/*
 * Minimal Cortex-M4 startup for STM32F411CEU6 (WeAct Black Pill).
 *
 * Boot sequence:
 *   1. Enable FPU (must happen before any FP instruction).
 *   2. Copy .data from flash to RAM, zero .bss.
 *   3. Configure the clock tree: HSE 25 MHz -> PLL -> 96 MHz SYSCLK.
 *   4. Jump to main().
 *
 * No interrupts other than Reset are wired up -- USART I/O is polled and
 * inference is a straight-line CPU loop. Any unexpected fault lands in
 * Default_Handler, an infinite loop that keeps state around for the debugger.
 */

#include <stdint.h>

extern uint32_t _sidata, _sdata, _edata, _sbss, _ebss, _estack;
extern int main(void);

/* -------- Register block addresses ----------------------------------- */
#define RCC_BASE        0x40023800UL
#define FLASH_BASE_R    0x40023C00UL
#define SCB_CPACR       (*(volatile uint32_t *)0xE000ED88UL)

#define REG32(addr)     (*(volatile uint32_t *)(addr))

#define RCC_CR          REG32(RCC_BASE + 0x00)
#define RCC_PLLCFGR     REG32(RCC_BASE + 0x04)
#define RCC_CFGR        REG32(RCC_BASE + 0x08)
#define FLASH_ACR       REG32(FLASH_BASE_R + 0x00)

#define RCC_CR_HSEON    (1U << 16)
#define RCC_CR_HSERDY   (1U << 17)
#define RCC_CR_PLLON    (1U << 24)
#define RCC_CR_PLLRDY   (1U << 25)

/*
 * Clock config:
 *   HSE = 25 MHz (WeAct Black Pill crystal)
 *   VCO in  = HSE / M = 25 / 25 = 1 MHz
 *   VCO out = 1 * N = 192 MHz
 *   SYSCLK  = VCO / P = 192 / 2 = 96 MHz
 *   AHB     = 96 MHz (/1)
 *   APB1    = 48 MHz (/2, max 50)
 *   APB2    = 96 MHz (/1, max 100)
 */
static void clock_init(void) {
    /* Kick the HSE on and wait for stability. */
    RCC_CR |= RCC_CR_HSEON;
    while (!(RCC_CR & RCC_CR_HSERDY)) { }

    /* Flash access: 3 wait states at 96 MHz, plus prefetch/caches. */
    FLASH_ACR = (1U << 8) | (1U << 9) | (1U << 10) | 3U;

    /* PLLM=25, PLLN=192, PLLP=0 (means /2), PLLSRC=HSE (bit 22). */
    RCC_PLLCFGR = 25U
                | (192U << 6)
                | (0U   << 16)
                | (1U   << 22)
                | (4U   << 24);   /* PLLQ=4, harmless (USB unused) */

    /* Bus prescalers before switching SW to PLL. */
    RCC_CFGR = (0U << 4)          /* HPRE  = /1 */
             | (0b100U << 10)     /* PPRE1 = /2 */
             | (0U << 13);        /* PPRE2 = /1 */

    /* Turn PLL on and wait. */
    RCC_CR |= RCC_CR_PLLON;
    while (!(RCC_CR & RCC_CR_PLLRDY)) { }

    /* Switch SYSCLK source to PLL. */
    RCC_CFGR |= 0b10U;                    /* SW = PLL */
    while (((RCC_CFGR >> 2) & 0b11U) != 0b10U) { }
}

static void Default_Handler(void);

void Reset_Handler(void) {
    /* Enable CP10 + CP11 (FPU) full access before touching any FP code. */
    SCB_CPACR |= (0xFU << 20);
    __asm__ volatile ("dsb; isb");

    /* Copy initialized data from flash to RAM. */
    uint32_t *src = &_sidata;
    for (uint32_t *dst = &_sdata; dst < &_edata; ) *dst++ = *src++;

    /* Zero the BSS. */
    for (uint32_t *dst = &_sbss; dst < &_ebss; ) *dst++ = 0;

    clock_init();
    (void)main();
    while (1) { }
}

static void Default_Handler(void) {
    while (1) { }
}

/* Minimal libc plumbing (nano.specs). We never actually call malloc, but
 * newlib pulls _sbrk in via reentrancy. Return -1 to make any accidental
 * allocation fail loudly rather than silently corrupting SRAM. */
void *_sbrk(int incr) { (void)incr; return (void *)-1; }
void _exit(int code)  { (void)code; while (1) { } }
int _write(int fd, char *buf, int n) { (void)fd; (void)buf; return n; }
int _read(int fd, char *buf, int n)  { (void)fd; (void)buf; (void)n; return 0; }
int _close(int fd)                   { (void)fd; return -1; }
int _lseek(int fd, int off, int w)   { (void)fd; (void)off; (void)w; return 0; }
int _fstat(int fd, void *st)         { (void)fd; (void)st; return 0; }
int _isatty(int fd)                  { (void)fd; return 1; }
int _getpid(void)                    { return 1; }
int _kill(int pid, int sig)          { (void)pid; (void)sig; return -1; }

/* -------- Vector table ------------------------------------------------ */
typedef void (*isr_t)(void);

__attribute__((section(".isr_vector"), used))
const isr_t vector_table[] = {
    (isr_t)&_estack,        /*  0: Initial stack pointer */
    Reset_Handler,          /*  1: Reset */
    Default_Handler,        /*  2: NMI */
    Default_Handler,        /*  3: HardFault */
    Default_Handler,        /*  4: MemManage */
    Default_Handler,        /*  5: BusFault */
    Default_Handler,        /*  6: UsageFault */
    0, 0, 0, 0,             /*  7-10: Reserved */
    Default_Handler,        /* 11: SVCall */
    Default_Handler,        /* 12: DebugMon */
    0,                      /* 13: Reserved */
    Default_Handler,        /* 14: PendSV */
    Default_Handler,        /* 15: SysTick */
    /* External IRQs left off -- polled I/O only. */
};
