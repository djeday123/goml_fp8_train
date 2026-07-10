package train

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/djeday123/goml/nn"
	"github.com/djeday123/goml/ops"
	"github.com/djeday123/goml/optim"
	"github.com/djeday123/goml/tensor"
	"github.com/djeday123/goml/tokenizer"
)

// TrainConfig holds training hyperparameters.
type TrainConfig struct {
	BatchSize   int
	SeqLen      int
	LR          float64
	WarmupSteps int
	TotalSteps  int
	MinLR       float64
	LogEvery    int
	EvalEvery   int
	GenEvery    int
	GenLen      int
	Temperature float32
}

func DefaultTrainConfig() TrainConfig {
	return TrainConfig{
		BatchSize:   4,
		SeqLen:      64,
		LR:          3e-4,
		WarmupSteps: 100,
		TotalSteps:  2000,
		MinLR:       3e-5,
		LogEvery:    10,
		EvalEvery:   100,
		GenEvery:    200,
		GenLen:      128,
		Temperature: 0.8,
	}
}

// Trainer handles the training loop.
type Trainer struct {
	Model     *nn.LLM
	Optimizer *optim.AdamW
	Tokenizer *tokenizer.ByteTokenizer
	Config    TrainConfig

	trainTokens []int64
	evalTokens  []int64
}

func NewTrainer(model *nn.LLM, text string, cfg TrainConfig) *Trainer {
	tok := tokenizer.NewByteTokenizer()
	allTokens := tok.Encode(text)

	splitIdx := int(float64(len(allTokens)) * 0.9)
	trainTokens := allTokens[:splitIdx]
	evalTokens := allTokens[splitIdx:]

	optimizer := optim.NewAdamW(model.Parameters(), cfg.LR)

	return &Trainer{
		Model:       model,
		Optimizer:   optimizer,
		Tokenizer:   tok,
		Config:      cfg,
		trainTokens: trainTokens,
		evalTokens:  evalTokens,
	}
}

func (t *Trainer) Train() {
	cfg := t.Config

	fmt.Printf("Training data: %d tokens (train: %d, eval: %d)\n",
		len(t.trainTokens)+len(t.evalTokens), len(t.trainTokens), len(t.evalTokens))
	fmt.Printf("Model: %d parameters\n", t.Model.CountParameters())
	fmt.Printf("Config: batch=%d, seqLen=%d, lr=%.1e, steps=%d\n\n",
		cfg.BatchSize, cfg.SeqLen, cfg.LR, cfg.TotalSteps)

	totalStart := time.Now()
	smoothLoss := float64(0)
	bestEvalLoss := math.MaxFloat64

	for step := 1; step <= cfg.TotalSteps; step++ {
		stepStart := time.Now()

		lr := optim.CosineSchedule(step, cfg.WarmupSteps, cfg.TotalSteps, cfg.LR, cfg.MinLR)
		t.Optimizer.SetLR(lr)

		inputs, targets := t.getBatch(t.trainTokens)

		logits, cache, err := t.Model.ForwardWithCache(inputs)
		if err != nil {
			fmt.Printf("Step %d forward error: %v\n", step, err)
			continue
		}

		loss, err := ops.CrossEntropyLoss(logits, targets)
		if err != nil {
			fmt.Printf("Step %d loss error: %v\n", step, err)
			continue
		}
		lossVal := float64(loss.ToFloat32Slice()[0])

		if smoothLoss == 0 {
			smoothLoss = lossVal
		} else {
			smoothLoss = 0.95*smoothLoss + 0.05*lossVal
		}

		t.Optimizer.ZeroGrad()
		dLogits, err := ops.CrossEntropyBackward(logits, targets)
		if err != nil {
			fmt.Printf("Step %d backward error: %v\n", step, err)
			continue
		}

		err = t.Model.Backward(cache, dLogits)
		if err != nil {
			fmt.Printf("Step %d model backward error: %v\n", step, err)
			continue
		}

		t.Optimizer.Step()

		stepTime := time.Since(stepStart)

		if step%cfg.LogEvery == 0 {
			tokPerSec := float64(cfg.BatchSize*cfg.SeqLen) / stepTime.Seconds()
			fmt.Printf("step %4d | loss %.4f (smooth %.4f) | lr %.2e | %.0f tok/s | %v\n",
				step, lossVal, smoothLoss, lr, tokPerSec, stepTime)
		}

		if step%cfg.EvalEvery == 0 {
			evalLoss := t.evaluate()
			improved := ""
			if evalLoss < bestEvalLoss {
				bestEvalLoss = evalLoss
				improved = " * best"
			}
			fmt.Printf("         -> eval loss: %.4f%s\n", evalLoss, improved)
		}

		if step%cfg.GenEvery == 0 {
			sample := t.generate("The ", cfg.GenLen, cfg.Temperature)
			fmt.Printf("         -> sample: %q\n", truncate(sample, 120))
		}
	}

	totalTime := time.Since(totalStart)
	fmt.Printf("\nTraining complete in %v\n", totalTime)
	fmt.Printf("Best eval loss: %.4f\n", bestEvalLoss)

	fmt.Println("\n--- Final generation samples ---")
	prompts := []string{"To be or ", "The king ", "What is ", "In the "}
	for _, p := range prompts {
		sample := t.generate(p, 200, 0.7)
		fmt.Printf("\nPrompt: %q\n-> %s\n", p, truncate(sample, 200))
	}
}

func (t *Trainer) getBatch(tokens []int64) (*tensor.Tensor, *tensor.Tensor) {
	cfg := t.Config
	maxStart := len(tokens) - cfg.SeqLen - 1

	inputData := make([]int64, cfg.BatchSize*cfg.SeqLen)
	targetData := make([]int64, cfg.BatchSize*cfg.SeqLen)

	for b := 0; b < cfg.BatchSize; b++ {
		start := rand.Intn(maxStart)
		for s := 0; s < cfg.SeqLen; s++ {
			inputData[b*cfg.SeqLen+s] = tokens[start+s]
			targetData[b*cfg.SeqLen+s] = tokens[start+s+1]
		}
	}

	inputs, _ := tensor.FromSlice(inputData, tensor.Shape{cfg.BatchSize, cfg.SeqLen})
	targets, _ := tensor.FromSlice(targetData, tensor.Shape{cfg.BatchSize, cfg.SeqLen})
	return inputs, targets
}

func (t *Trainer) evaluate() float64 {
	numBatches := 5
	totalLoss := float64(0)

	for i := 0; i < numBatches; i++ {
		inputs, targets := t.getBatch(t.evalTokens)
		logits, _, err := t.Model.ForwardWithCache(inputs)
		if err != nil {
			continue
		}
		loss, err := ops.CrossEntropyLoss(logits, targets)
		if err != nil {
			continue
		}
		totalLoss += float64(loss.ToFloat32Slice()[0])
	}

	return totalLoss / float64(numBatches)
}

func (t *Trainer) generate(prompt string, maxLen int, temperature float32) string {
	tokens := t.Tokenizer.Encode(prompt)
	cfg := t.Model.Config

	for i := 0; i < maxLen; i++ {
		start := 0
		if len(tokens) > cfg.MaxSeqLen {
			start = len(tokens) - cfg.MaxSeqLen
		}
		window := tokens[start:]

		input, _ := tensor.FromSlice(window, tensor.Shape{1, len(window)})
		logits, _ := t.Model.Forward(input)

		logitsData := logits.ToFloat32Slice()
		lastStart := (len(window) - 1) * cfg.VocabSize
		lastLogits, _ := tensor.FromSlice(
			logitsData[lastStart:lastStart+cfg.VocabSize],
			tensor.Shape{cfg.VocabSize},
		)

		nextToken := int64(nn.TopKSample(lastLogits, 40, temperature))
		tokens = append(tokens, nextToken)
	}

	return t.Tokenizer.Decode(tokens)
}

func truncate(s string, maxLen int) string {
	if len(s) > maxLen {
		return s[:maxLen] + "..."
	}
	return s
}
