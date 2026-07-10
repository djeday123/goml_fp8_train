/*
 * fa_dispatch.cpp — pure-function kernel selector.
 *
 * Single table = single source of truth. Easy to attach an autotuner later
 * (replace table lookup with measured-time cache).
 *
 * Rules derived from production bench data (memory: forward-final-champion-numbers,
 * v117b/v118b porting, v122 experiment, dispatcher audit).
 */
#include "../include/fa_sm120.h"
#include <stdio.h>
#include <stdlib.h>

extern "C" fa_kernel_id_t fa_dispatch_select(int bh, int sl, int hd, int causal, int window)
{
    if (hd != 64 && hd != 128) return FA_KERNEL_NONE;
    if (bh <= 0 || sl <= 0) return FA_KERNEL_NONE;

    fa_kernel_id_t chosen = FA_KERNEL_NONE;

    if (hd == 64) {
        int br = 128;
        int grid = bh * ((sl + br - 1) / br);
        chosen = (grid > 128) ? FA_KERNEL_V89 : FA_KERNEL_V80B;
    } else {
        /* hd == 128 */
        int wnd = (window > 0) ? 1 : 0;

        if (wnd) {
            /* Window niches first (tight by config). */
            if (bh == 4 && sl == 8192 && window == 1024) {
                chosen = FA_KERNEL_V117B;
            } else if (bh == 4 && sl == 4096 && window == 1024) {
                chosen = FA_KERNEL_V118;
            } else if (bh == 8 && sl == 8192 && window == 1024) {
                chosen = FA_KERNEL_V118;
            } else {
                /* Larger grid with window — v121 (window champion). */
                chosen = FA_KERNEL_V121;
            }
        } else {
            /* wnd=0 (full attention or causal-no-window). */
            if (bh == 4 && sl <= 2048) {
                chosen = FA_KERNEL_V122;
            } else if (bh == 4 && sl == 4096) {
                chosen = FA_KERNEL_V118;
            } else if (bh == 8 && sl <= 4096) {
                chosen = FA_KERNEL_V118;
            } else if (bh == 16 && sl == 2048) {
                chosen = FA_KERNEL_V118;
            } else if ((bh == 16 && sl == 4096) ||
                       (bh == 32 && sl == 2048) ||
                       (bh == 128 && sl == 2048) ||
                       (bh == 64 && sl == 4096)) {
                /* Narrow boundary configs where v96b ≥ v121r empirically. */
                chosen = FA_KERNEL_V96B;
            } else {
                /* Peak target (largest portion of dispatcher). */
                chosen = FA_KERNEL_V121R;
            }
        }
    }

    static int debug = -1;
    if (debug < 0) {
        const char* e = getenv("FA_SM120_DEBUG");
        debug = (e && *e == '1') ? 1 : 0;
    }
    if (debug) {
        fprintf(stderr,
                "[fa_sm120] dispatch bh=%d sl=%d hd=%d causal=%d wnd=%d -> %s\n",
                bh, sl, hd, causal, window, fa_kernel_name(chosen));
    }
    return chosen;
}

extern "C" const char* fa_kernel_name(fa_kernel_id_t id)
{
    switch (id) {
    case FA_KERNEL_V121R: return "v121r";
    case FA_KERNEL_V121:  return "v121";
    case FA_KERNEL_V118:  return "v118";
    case FA_KERNEL_V122:  return "v122";
    case FA_KERNEL_V117B: return "v117b";
    case FA_KERNEL_V96B:  return "v96b";
    case FA_KERNEL_V89:   return "v89";
    case FA_KERNEL_V80B:  return "v80b";
    case FA_KERNEL_NONE:
    default:              return "<none>";
    }
}
