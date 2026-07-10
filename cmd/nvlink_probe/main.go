// nvlink_probe.go — Прямой доступ к NVLink через PCIe/MMIO
// БЕЗ CGO, БЕЗ NVIDIA библиотек. Чистый Go + syscall.
//
// sudo go run nvlink_probe.go

package main

import (
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"unsafe"
)

// ============================================================================
// Известные MMIO offsets для NVLink (из envytools / nouveau / open-gpu-kernel)
// ============================================================================

const (
	NV_PMC_BOOT_0   = 0x00000000
	NV_PMC_ENABLE   = 0x00000200
	NV_FUSE_START   = 0x00021000
	NV_FUSE_END     = 0x00022000
	NV_FUSE_NVLINK  = 0x00021C00 // NVLink disable fuse region (Ampere+)
	NV_PTOP_DEVINFO = 0x00022700
)

type NvLinkBlock struct {
	Start uint32
	End   uint32
	Name  string
}

var nvlinkBlocks = []NvLinkBlock{
	{0x00A00000, 0x00A10000, "IOCTRL (NVLink IO Controller)"},
	{0x00A20000, 0x00A30000, "NVLIPT (NVLink IP Topology)"},
	{0x01140000, 0x01180000, "MINION (NVLink Firmware Controller)"},
	{0x01180000, 0x011A0000, "NVLW (NVLink Wrapper)"},
	{0x011A0000, 0x011C0000, "NVLTLC (NVLink Transport Layer)"},
	{0x01300000, 0x01400000, "NVLPHY (NVLink PHY/SerDes)"},
}

// ============================================================================
// GPU Discovery через sysfs
// ============================================================================

type GPU struct {
	BDF      string
	DeviceID uint16
	SysPath  string
	BAR0Addr uint64
	BAR0Size uint64
}

func findGPUs() []GPU {
	var gpus []GPU

	matches, _ := filepath.Glob("/sys/bus/pci/devices/*/vendor")
	for _, vpath := range matches {
		data, err := os.ReadFile(vpath)
		if err != nil {
			continue
		}
		vendor := strings.TrimSpace(string(data))
		if vendor != "0x10de" {
			continue
		}

		dir := filepath.Dir(vpath)
		bdf := filepath.Base(dir)

		// Device ID
		var devID uint16
		if d, err := os.ReadFile(filepath.Join(dir, "device")); err == nil {
			fmt.Sscanf(strings.TrimSpace(string(d)), "0x%x", &devID)
		}

		// Class check - только GPU (0x0300xx)
		if d, err := os.ReadFile(filepath.Join(dir, "class")); err == nil {
			var class uint32
			fmt.Sscanf(strings.TrimSpace(string(d)), "0x%x", &class)
			if (class >> 16) != 0x03 {
				continue
			}
		}

		// BAR0 из resource
		var bar0Addr, bar0Size uint64
		if d, err := os.ReadFile(filepath.Join(dir, "resource")); err == nil {
			lines := strings.Split(string(d), "\n")
			if len(lines) > 0 {
				var start, end uint64
				fmt.Sscanf(lines[0], "0x%x 0x%x", &start, &end)
				bar0Addr = start
				bar0Size = end - start + 1
			}
		}

		gpus = append(gpus, GPU{
			BDF:      bdf,
			DeviceID: devID,
			SysPath:  dir,
			BAR0Addr: bar0Addr,
			BAR0Size: bar0Size,
		})
	}
	return gpus
}

// ============================================================================
// BAR0 MMIO mapping — прямой mmap через syscall
// ============================================================================

type MMIO struct {
	data []byte
	size uint64
}

func mapBAR0(gpu *GPU) (*MMIO, error) {
	path := filepath.Join(gpu.SysPath, "resource0")

	fd, err := syscall.Open(path, syscall.O_RDONLY|syscall.O_SYNC, 0)
	if err != nil {
		return nil, fmt.Errorf("open %s: %v (нужен root)", path, err)
	}
	defer syscall.Close(fd)

	size := gpu.BAR0Size
	if size == 0 || size > 256*1024*1024 {
		size = 32 * 1024 * 1024
	}

	data, err := syscall.Mmap(fd, 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("mmap BAR0: %v", err)
	}

	return &MMIO{data: data, size: size}, nil
}

func (m *MMIO) Close() {
	syscall.Munmap(m.data)
}

func (m *MMIO) Read32(offset uint32) uint32 {
	if uint64(offset)+4 > m.size {
		return 0xDEADDEAD
	}
	return *(*uint32)(unsafe.Pointer(&m.data[offset]))
}

// ============================================================================
// PCIe config space
// ============================================================================

func readPCIeConfig(gpu *GPU) {
	path := filepath.Join(gpu.SysPath, "config")
	data, err := os.ReadFile(path)
	if err != nil {
		fmt.Printf("  [!] PCIe config: %v\n", err)
		return
	}

	fmt.Printf("  PCIe Config: %d bytes\n", len(data))

	// Capabilities chain
	if len(data) > 0x34 {
		capPtr := data[0x34]
		fmt.Printf("  Capabilities:\n")
		for capPtr != 0 && capPtr != 0xFF && int(capPtr)+1 < len(data) {
			capID := data[capPtr]
			next := data[capPtr+1]

			name := "Unknown"
			switch capID {
			case 0x01:
				name = "PM"
			case 0x05:
				name = "MSI"
			case 0x09:
				name = "Vendor Specific"
			case 0x10:
				name = "PCIe"
			case 0x11:
				name = "MSI-X"
			}
			fmt.Printf("    [0x%02X] ID=0x%02X %s", capPtr, capID, name)

			if capID == 0x09 && int(capPtr)+20 < len(data) {
				fmt.Printf("\n           Data: ")
				for i := 0; i < 16 && int(capPtr)+4+i < len(data); i++ {
					fmt.Printf("%02X ", data[int(capPtr)+4+i])
				}
			}
			fmt.Println()
			capPtr = next
		}
	}

	// Extended capabilities
	if len(data) > 0x104 {
		fmt.Printf("  Extended Capabilities:\n")
		off := 0x100
		for off > 0 && off+4 <= len(data) {
			hdr := binary.LittleEndian.Uint32(data[off:])
			capID := hdr & 0xFFFF
			nextOff := int((hdr >> 20) & 0xFFC)
			if capID == 0 && nextOff == 0 {
				break
			}

			name := fmt.Sprintf("0x%04X", capID)
			switch capID {
			case 0x0001:
				name = "AER"
			case 0x000B:
				name = "Vendor Specific Ext"
			case 0x0019:
				name = "Secondary PCIe"
			case 0x0025:
				name = "Data Link Feature"
			}
			fmt.Printf("    [0x%03X] %s", off, name)

			if capID == 0x000B && off+36 <= len(data) {
				fmt.Printf("\n           NVIDIA data: ")
				for i := 0; i < 32 && off+4+i < len(data); i++ {
					fmt.Printf("%02X ", data[off+4+i])
				}
			}
			fmt.Println()

			off = nextOff
		}
	}
}

// ============================================================================
// FUSE — аппаратные предохранители
// ============================================================================

func readFuses(m *MMIO) {
	fmt.Println("\n  === FUSE Registers ===")

	boot0 := m.Read32(NV_PMC_BOOT_0)
	fmt.Printf("  BOOT_0 = 0x%08X\n", boot0)

	arch := (boot0 >> 20) & 0xFF
	impl := (boot0 >> 12) & 0xFF
	fmt.Printf("  Arch: 0x%02X Impl: 0x%02X", arch, impl)
	switch {
	case arch >= 0x19 && arch <= 0x19:
		fmt.Printf(" → Ada Lovelace")
	case arch == 0x17:
		fmt.Printf(" → Ampere")
	case arch == 0x18:
		fmt.Printf(" → Hopper")
	}
	fmt.Println()

	// Scan fuse region
	fmt.Printf("\n  FUSE scan [0x%05X - 0x%05X]:\n", NV_FUSE_START, NV_FUSE_END)
	for off := uint32(NV_FUSE_START); off < NV_FUSE_END; off += 4 {
		val := m.Read32(off)
		if val != 0 && val != 0xFFFFFFFF && val != 0xDEADDEAD {
			fmt.Printf("    [0x%05X] = 0x%08X", off, val)
			if off >= 0x21C00 && off < 0x21C40 {
				fmt.Printf("  ← NVLINK FUSE REGION")
				if val&1 != 0 {
					fmt.Printf(" [DISABLED by fuse]")
				} else {
					fmt.Printf(" [NOT disabled → silicon has NVLink capability]")
				}
			}
			fmt.Println()
		}
	}

	// PMC_ENABLE
	pmcEn := m.Read32(NV_PMC_ENABLE)
	fmt.Printf("\n  PMC_ENABLE = 0x%08X\n", pmcEn)
	fmt.Printf("  Bit 27: %d  Bit 28: %d  Bit 29: %d  (potential NVLink engine bits)\n",
		(pmcEn>>27)&1, (pmcEn>>28)&1, (pmcEn>>29)&1)
}

// ============================================================================
// NVLink MMIO block scan
// ============================================================================

func scanNvLinkBlocks(m *MMIO) {
	fmt.Println("\n  === NVLink MMIO Blocks ===")
	fmt.Println("  Живые регистры = железо подключено к внутренней шине GPU")

	for _, blk := range nvlinkBlocks {
		if uint64(blk.End) > m.size {
			fmt.Printf("\n  [%s] за пределами BAR0 (%dMB)\n", blk.Name, m.size/(1024*1024))
			continue
		}

		liveCount := 0
		deadCount := 0
		var firstLiveOff uint32
		var firstLiveVal uint32

		for off := blk.Start; off < blk.End; off += 256 {
			val := m.Read32(off)
			if val == 0 || val == 0xFFFFFFFF || val == 0xDEADDEAD ||
				val == 0xBADF1000 || val == 0xBADF5040 {
				deadCount++
			} else {
				liveCount++
				if firstLiveOff == 0 {
					firstLiveOff = off
					firstLiveVal = val
				}
			}
		}

		fmt.Printf("\n  [%s]\n", blk.Name)
		fmt.Printf("    0x%06X - 0x%06X | alive: %d dead: %d\n",
			blk.Start, blk.End, liveCount, deadCount)

		if liveCount > 0 {
			fmt.Printf("    >>> ЖИВЫЕ РЕГИСТРЫ — ЖЕЛЕЗО ПРИСУТСТВУЕТ <<<\n")
			fmt.Printf("    First: [0x%06X] = 0x%08X\n", firstLiveOff, firstLiveVal)

			// Дамп первых живых
			shown := 0
			for off := blk.Start; off < blk.End && shown < 12; off += 4 {
				val := m.Read32(off)
				if val != 0 && val != 0xFFFFFFFF && val != 0xDEADDEAD {
					fmt.Printf("    [0x%06X] = 0x%08X\n", off, val)
					shown++
				}
			}
		} else {
			fmt.Printf("    Мёртвый блок — не подключён или power-gated\n")
		}
	}
}

// ============================================================================
// PHY scan
// ============================================================================

func scanPHY(m *MMIO) {
	fmt.Println("\n  === NVLink PHY SerDes ===")

	phyBases := []uint32{
		0x01300000, 0x01300800, 0x01301000, 0x01301800,
		0x01302000, 0x01303000, 0x01304000, 0x01308000, 0x01310000,
	}

	livePhys := 0
	for _, base := range phyBases {
		if uint64(base)+256 > m.size {
			continue
		}
		alive := false
		for off := uint32(0); off < 256; off += 4 {
			val := m.Read32(base + off)
			if val != 0 && val != 0xFFFFFFFF && val != 0xDEADDEAD {
				alive = true
				break
			}
		}
		status := "dead"
		if alive {
			status = "ALIVE <<<<<"
			livePhys++
		}
		fmt.Printf("    PHY @ 0x%06X: %s\n", base, status)
	}

	fmt.Printf("    Живых PHY: %d\n", livePhys)
	if livePhys > 0 {
		fmt.Println("    >>> NVLink ТРАНСИВЕРЫ ПРИСУТСТВУЮТ В КРЕМНИИ <<<")
	}
}

// ============================================================================
// Wide scan — полный sweep NVLink региона
// ============================================================================

func wideScan(m *MMIO) {
	fmt.Println("\n  === Wide Scan: NVLink Region 0xA00000 - 0x1400000 ===")

	scanStart := uint32(0x00A00000)
	scanEnd := uint32(0x01400000)
	if uint64(scanEnd) > m.size {
		scanEnd = uint32(m.size)
	}
	if uint64(scanStart) >= m.size {
		fmt.Printf("  [!] BAR0 (%dMB) не покрывает NVLink регион\n", m.size/(1024*1024))
		return
	}

	livePages := 0
	totalPages := 0

	for page := scanStart; page < scanEnd; page += 4096 {
		totalPages++
		for off := uint32(0); off < 4096; off += 256 {
			val := m.Read32(page + off)
			if val != 0 && val != 0xFFFFFFFF && val != 0xDEADDEAD &&
				val != 0xBADF1000 && val != 0xBADF5040 {
				livePages++
				if livePages <= 15 {
					fmt.Printf("    LIVE @ 0x%06X = 0x%08X\n", page+off, val)
				}
				break
			}
		}
	}

	fmt.Printf("\n    Scanned: %d pages, Live: %d (%.1f%%)\n",
		totalPages, livePages, 100.0*float64(livePages)/float64(totalPages))

	if livePages > 0 {
		fmt.Println("    >>> NVLink MMIO РЕГИОН СОДЕРЖИТ ЖИВЫЕ ДАННЫЕ <<<")
	} else {
		fmt.Println("    Регион мёртв: power-gated, fused-off, или BAR0 слишком маленький")
	}
}

// ============================================================================
// Device Info Table
// ============================================================================

func readDeviceInfo(m *MMIO) {
	fmt.Println("\n  === PTOP Device Info ===")

	base := uint32(NV_PTOP_DEVINFO)
	if uint64(base)+256 > m.size {
		fmt.Println("  За пределами BAR0")
		return
	}

	for i := 0; i < 64; i++ {
		val := m.Read32(base + uint32(i*4))
		if val == 0 || val == 0xFFFFFFFF {
			continue
		}

		devType := (val >> 2) & 0x3F
		inst := (val >> 8) & 0xFF

		fmt.Printf("    Entry %2d: 0x%08X type=%d inst=%d", i, val, devType, inst)
		switch devType {
		case 0x01:
			fmt.Printf(" → GR")
		case 0x03:
			fmt.Printf(" → CE")
		case 0x0E:
			fmt.Printf(" → NVENC")
		case 0x0F:
			fmt.Printf(" → NVDEC")
		case 0x13:
			fmt.Printf(" → *** NVLINK ***")
		}
		fmt.Println()
	}
}

// ============================================================================
// Main
// ============================================================================

func main() {
	fmt.Println("╔═══════════════════════════════════════════════════════════╗")
	fmt.Println("║  NVLink Probe — В ОБХОД CUDA/NVML (Pure Go + syscall)   ║")
	fmt.Println("║  Прямой PCIe config + BAR0 MMIO + FUSE registers        ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════╝")
	fmt.Println()

	if os.Geteuid() != 0 {
		fmt.Println("[!] Нужен root для BAR0 MMIO")
		fmt.Println("[!] sudo go run nvlink_probe.go")
		fmt.Println()
	}

	gpus := findGPUs()
	if len(gpus) == 0 {
		fmt.Println("NVIDIA GPU не найдены")
		os.Exit(1)
	}

	fmt.Printf("Найдено %d GPU\n\n", len(gpus))

	for i, gpu := range gpus {
		fmt.Printf("═══ GPU %d: %s (0x%04X) BAR0: %dMB ═══\n\n",
			i, gpu.BDF, gpu.DeviceID, gpu.BAR0Size/(1024*1024))

		// PCIe config
		readPCIeConfig(&gpu)

		// BAR0 MMIO
		mmio, err := mapBAR0(&gpu)
		if err != nil {
			fmt.Printf("  [!] BAR0: %v\n", err)
			continue
		}
		defer mmio.Close()

		fmt.Printf("  BAR0 mapped: %d MB\n", mmio.size/(1024*1024))

		readFuses(mmio)
		readDeviceInfo(mmio)
		scanNvLinkBlocks(mmio)
		scanPHY(mmio)
		wideScan(mmio)
	}

	fmt.Println()
	fmt.Println("═══ ИНТЕРПРЕТАЦИЯ ═══")
	fmt.Println()
	fmt.Println("Сценарий A: Fuse=0, MMIO живой, PHY живой")
	fmt.Println("  → NVLink есть в кремнии и НЕ отключён фьюзами")
	fmt.Println("  → NVIDIA скрывает через драйвер/NVML")
	fmt.Println("  → Вопрос только в PCB разводке и коннекторе")
	fmt.Println()
	fmt.Println("Сценарий B: Fuse=0, MMIO мёртвый")
	fmt.Println("  → Кремний имеет NVLink, но NVIDIA power-gates блок")
	fmt.Println("  → Программное отключение при живом железе")
	fmt.Println()
	fmt.Println("Сценарий C: Fuse=1")
	fmt.Println("  → NVLink прожжён на фабрике — необратимо отключён")
	fmt.Println()
	fmt.Println("Сценарий D: BAR0 не покрывает NVLink регион")
	fmt.Println("  → Попробуй resizable BAR или прямой /dev/mem доступ")
}
