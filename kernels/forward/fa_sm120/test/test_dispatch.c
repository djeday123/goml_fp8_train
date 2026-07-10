/*
 * test_dispatch.c — фиксирует диспетчер-таблицу тестом.
 * Если поменяли диспетчер — этот тест обязан соответственно обновиться.
 * Build: gcc -I../include -o test_dispatch test_dispatch.c -L.. -lfa_sm120
 */
#include "fa_sm120.h"
#include <stdio.h>
#include <string.h>

struct Case { int bh, sl, hd, causal, wnd; fa_kernel_id_t expect; const char* note; };

int main(void)
{
    struct Case cases[] = {
        /* hd=128 wnd=0 — peak target и niches */
        {  4, 1024, 128, 0,    0, FA_KERNEL_V122,  "bh=4 sl=1024 wave-tail" },
        {  4, 2048, 128, 0,    0, FA_KERNEL_V122,  "bh=4 sl=2048 wave-tail" },
        {  4, 4096, 128, 0,    0, FA_KERNEL_V118,  "bh=4 sl=4096 mid" },
        {  8, 2048, 128, 0,    0, FA_KERNEL_V118,  "bh=8 sl=2048 mid" },
        {  8, 4096, 128, 0,    0, FA_KERNEL_V118,  "bh=8 sl=4096 mid" },
        { 16, 2048, 128, 0,    0, FA_KERNEL_V118,  "bh=16 sl=2048 mid" },
        { 16, 4096, 128, 0,    0, FA_KERNEL_V96B,  "narrow boundary" },
        { 32, 2048, 128, 0,    0, FA_KERNEL_V96B,  "narrow boundary" },
        { 32, 4096, 128, 0,    0, FA_KERNEL_V121R, "peak" },
        { 32, 8192, 128, 0,    0, FA_KERNEL_V121R, "peak" },
        { 64, 4096, 128, 0,    0, FA_KERNEL_V96B,  "narrow boundary" },
        { 64, 8192, 128, 0,    0, FA_KERNEL_V121R, "PRIMARY peak target" },
        {128, 2048, 128, 0,    0, FA_KERNEL_V96B,  "narrow boundary" },
        {128, 4096, 128, 0,    0, FA_KERNEL_V121R, "peak" },
        {128, 8192, 128, 0,    0, FA_KERNEL_V121R, "SECONDARY peak target" },
        {256, 2048, 128, 0,    0, FA_KERNEL_V121R, "peak" },
        {256, 4096, 128, 0,    0, FA_KERNEL_V121R, "peak" },
        /* hd=128 window */
        {  4, 4096, 128, 1, 1024, FA_KERNEL_V118,  "bh=4 wnd=1024 mid niche" },
        {  4, 8192, 128, 1, 1024, FA_KERNEL_V117B, "bh=4 sl=8192 wnd=1024 niche" },
        {  8, 8192, 128, 1, 1024, FA_KERNEL_V118,  "bh=8 wnd=1024 niche" },
        { 16, 8192, 128, 1, 1024, FA_KERNEL_V121,  "window champion" },
        { 32, 8192, 128, 1, 1024, FA_KERNEL_V121,  "window champion" },
        { 64, 8192, 128, 1, 1024, FA_KERNEL_V121,  "window champion" },
        /* hd=64 */
        {  4, 1024,  64, 0,    0, FA_KERNEL_V80B,  "hd=64 wave-tail" },
        {  4, 2048,  64, 0,    0, FA_KERNEL_V80B,  "hd=64 small grid" },
        { 64, 8192,  64, 0,    0, FA_KERNEL_V89,   "hd=64 peak" },
        { 16, 4096,  64, 0,    0, FA_KERNEL_V89,   "hd=64 peak" },
        /* Negative cases */
        { 64, 8192,  96, 0,    0, FA_KERNEL_NONE,  "hd=96 unsupported" },
        {  0,  100, 128, 0,    0, FA_KERNEL_NONE,  "bh=0 invalid" },
    };
    int n_cases = (int)(sizeof(cases) / sizeof(cases[0]));
    int n_fail = 0;
    for (int i = 0; i < n_cases; i++) {
        struct Case* c = &cases[i];
        fa_kernel_id_t got = fa_dispatch_select(c->bh, c->sl, c->hd, c->causal, c->wnd);
        const char* tag = (got == c->expect) ? "OK  " : "FAIL";
        if (got != c->expect) n_fail++;
        printf("  [%s] bh=%-3d sl=%-5d hd=%-3d ca=%d wnd=%-4d  expect=%-6s got=%-6s  %s\n",
               tag, c->bh, c->sl, c->hd, c->causal, c->wnd,
               fa_kernel_name(c->expect), fa_kernel_name(got), c->note);
    }
    printf("\n%d/%d cases pass.\n", n_cases - n_fail, n_cases);
    return (n_fail == 0) ? 0 : 1;
}
