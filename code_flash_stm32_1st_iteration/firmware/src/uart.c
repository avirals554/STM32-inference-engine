/*
 * Bare-metal USART1 driver for STM32F411 (WeAct Black Pill).
 *
 * Wiring:
 *   PA9  -> USART1 TX  (chip -> host)
 *   PA10 -> USART1 RX  (host -> chip)
 *
 * Clocks: PCLK2 = 96 MHz (see startup.c), 115200 baud, 8N1, no flow control.
 * Polled I/O -- no interrupts, no DMA, keeps the boot path trivial.
 */

#include "uart.h"

#include <stdint.h>

/* -------- Register block addresses ----------------------------------- */
#define RCC_BASE        0x40023800UL
#define GPIOA_BASE      0x40020000UL
#define USART1_BASE     0x40011000UL

#define REG32(addr)     (*(volatile uint32_t *)(addr))

/* RCC */
#define RCC_AHB1ENR     REG32(RCC_BASE + 0x30)
#define RCC_APB2ENR     REG32(RCC_BASE + 0x44)

/* GPIOA */
#define GPIOA_MODER     REG32(GPIOA_BASE + 0x00)
#define GPIOA_OSPEEDR   REG32(GPIOA_BASE + 0x08)
#define GPIOA_PUPDR     REG32(GPIOA_BASE + 0x0C)
#define GPIOA_AFRH      REG32(GPIOA_BASE + 0x24)

/* USART1 */
#define USART1_SR       REG32(USART1_BASE + 0x00)
#define USART1_DR       REG32(USART1_BASE + 0x04)
#define USART1_BRR      REG32(USART1_BASE + 0x08)
#define USART1_CR1      REG32(USART1_BASE + 0x0C)

#define USART_SR_TXE    (1U << 7)
#define USART_SR_RXNE   (1U << 5)
#define USART_CR1_UE    (1U << 13)
#define USART_CR1_TE    (1U << 3)
#define USART_CR1_RE    (1U << 2)

void uart_init(void) {
    /* Enable GPIOA and USART1 clocks. */
    RCC_AHB1ENR |= (1U << 0);  /* GPIOAEN */
    RCC_APB2ENR |= (1U << 4);  /* USART1EN */

    /* PA9, PA10 -> alternate function mode (MODER = 0b10). */
    uint32_t moder = GPIOA_MODER;
    moder &= ~((0x3U << (9 * 2)) | (0x3U << (10 * 2)));
    moder |=  ((0x2U << (9 * 2)) | (0x2U << (10 * 2)));
    GPIOA_MODER = moder;

    /* High speed for TX, PU on RX to avoid noise when host disconnected. */
    GPIOA_OSPEEDR |= (0x3U << (9 * 2));
    GPIOA_PUPDR   = (GPIOA_PUPDR & ~(0x3U << (10 * 2))) | (0x1U << (10 * 2));

    /* AF7 for USART1 on PA9 and PA10 (AFRH covers pins 8..15). */
    uint32_t afrh = GPIOA_AFRH;
    afrh &= ~((0xFU << ((9 - 8) * 4)) | (0xFU << ((10 - 8) * 4)));
    afrh |=  ((0x7U << ((9 - 8) * 4)) | (0x7U << ((10 - 8) * 4)));
    GPIOA_AFRH = afrh;

    /* BRR = round(PCLK2 / baud). At 96 MHz / 115200 = 833.
     * mantissa=52, fraction=1 -> encoded value 833, ~0.04% error. */
    USART1_BRR = 833;

    /* 8N1 default, oversampling 16, enable TX/RX and USART. */
    USART1_CR1 = USART_CR1_UE | USART_CR1_TE | USART_CR1_RE;
}

void uart_putc(char c) {
    while (!(USART1_SR & USART_SR_TXE)) { }
    USART1_DR = (uint8_t)c;
}

void uart_puts(const char *s) {
    while (*s) uart_putc(*s++);
}

void uart_write(const char *buf, size_t n) {
    for (size_t i = 0; i < n; i++) uart_putc(buf[i]);
}

uint8_t uart_getc(void) {
    while (!(USART1_SR & USART_SR_RXNE)) { }
    return (uint8_t)USART1_DR;
}
