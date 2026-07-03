/*
 * Bare-metal I2C1 master driver.
 *
 * Wiring (WeAct Black Pill):
 *   PB6 -> SCL   (open-drain, pull-up on OLED module or wire a 4k7)
 *   PB7 -> SDA
 *
 * Runs in fast mode (400 kHz) on APB1 = 48 MHz. Only master-transmit is
 * implemented; the SSD1306 is write-only in our usage.
 */

#include "i2c.h"

#include <stdint.h>

#define RCC_BASE       0x40023800UL
#define GPIOB_BASE     0x40020400UL
#define I2C1_BASE      0x40005400UL
#define REG32(addr)    (*(volatile uint32_t *)(addr))

#define RCC_AHB1ENR    REG32(RCC_BASE + 0x30)
#define RCC_APB1ENR    REG32(RCC_BASE + 0x40)

#define GPIOB_MODER    REG32(GPIOB_BASE + 0x00)
#define GPIOB_OTYPER   REG32(GPIOB_BASE + 0x04)
#define GPIOB_OSPEEDR  REG32(GPIOB_BASE + 0x08)
#define GPIOB_PUPDR    REG32(GPIOB_BASE + 0x0C)
#define GPIOB_AFRL     REG32(GPIOB_BASE + 0x20)

#define I2C1_CR1       REG32(I2C1_BASE + 0x00)
#define I2C1_CR2       REG32(I2C1_BASE + 0x04)
#define I2C1_DR        REG32(I2C1_BASE + 0x10)
#define I2C1_SR1       REG32(I2C1_BASE + 0x14)
#define I2C1_SR2       REG32(I2C1_BASE + 0x18)
#define I2C1_CCR       REG32(I2C1_BASE + 0x1C)
#define I2C1_TRISE     REG32(I2C1_BASE + 0x20)

#define I2C_CR1_PE     (1U << 0)
#define I2C_CR1_START  (1U << 8)
#define I2C_CR1_STOP   (1U << 9)
#define I2C_CR1_SWRST  (1U << 15)

#define I2C_SR1_SB     (1U << 0)
#define I2C_SR1_ADDR   (1U << 1)
#define I2C_SR1_BTF    (1U << 2)
#define I2C_SR1_TXE    (1U << 7)
#define I2C_SR1_AF     (1U << 10)  /* ACK failure */

#define I2C_TIMEOUT    200000

void i2c1_init(void) {
    RCC_AHB1ENR |= (1U << 1);   /* GPIOBEN */
    RCC_APB1ENR |= (1U << 21);  /* I2C1EN  */

    /* PB6, PB7 -> AF mode, open-drain, high speed, pull-up. */
    uint32_t moder = GPIOB_MODER;
    moder &= ~((0x3U << (6 * 2)) | (0x3U << (7 * 2)));
    moder |=  ((0x2U << (6 * 2)) | (0x2U << (7 * 2)));
    GPIOB_MODER = moder;

    GPIOB_OTYPER  |= (1U << 6) | (1U << 7);
    GPIOB_OSPEEDR |= (0x3U << (6 * 2)) | (0x3U << (7 * 2));

    uint32_t pupdr = GPIOB_PUPDR;
    pupdr &= ~((0x3U << (6 * 2)) | (0x3U << (7 * 2)));
    pupdr |=  ((0x1U << (6 * 2)) | (0x1U << (7 * 2)));  /* pull-up */
    GPIOB_PUPDR = pupdr;

    /* AF4 on PB6, PB7 (AFRL covers pins 0..7). */
    uint32_t afrl = GPIOB_AFRL;
    afrl &= ~((0xFU << (6 * 4)) | (0xFU << (7 * 4)));
    afrl |=  ((0x4U << (6 * 4)) | (0x4U << (7 * 4)));
    GPIOB_AFRL = afrl;

    /* Reset then configure. */
    I2C1_CR1 = I2C_CR1_SWRST;
    I2C1_CR1 = 0;

    I2C1_CR2   = 48;                        /* APB1 in MHz */
    /* Fast mode 400 kHz: CCR = APB1 / (3 * baud) = 48e6 / 1.2e6 = 40, FS bit. */
    I2C1_CCR   = (1U << 15) | 40;
    /* TRISE for fast mode: max rise = 300 ns => 300ns * 48MHz + 1 = 15. */
    I2C1_TRISE = 15;
    I2C1_CR1   = I2C_CR1_PE;
}

static int wait_sr1(uint32_t mask) {
    for (uint32_t i = 0; i < I2C_TIMEOUT; i++) {
        uint32_t s = I2C1_SR1;
        if (s & mask) return 1;
        if (s & I2C_SR1_AF) { I2C1_SR1 = ~I2C_SR1_AF; return 0; }
    }
    return 0;
}

int i2c1_write2(uint8_t addr7,
                const uint8_t *prefix, size_t plen,
                const uint8_t *data, size_t dlen) {
    I2C1_CR1 |= I2C_CR1_START;
    if (!wait_sr1(I2C_SR1_SB)) return -1;

    I2C1_DR = (uint32_t)(addr7 << 1);
    if (!wait_sr1(I2C_SR1_ADDR)) return -2;
    (void)I2C1_SR2;  /* clearing ADDR requires reading SR1 then SR2 */

    for (size_t i = 0; i < plen; i++) {
        if (!wait_sr1(I2C_SR1_TXE)) return -3;
        I2C1_DR = prefix[i];
    }
    for (size_t i = 0; i < dlen; i++) {
        if (!wait_sr1(I2C_SR1_TXE)) return -4;
        I2C1_DR = data[i];
    }
    if (!wait_sr1(I2C_SR1_BTF)) return -5;

    I2C1_CR1 |= I2C_CR1_STOP;
    return 0;
}

int i2c1_write(uint8_t addr7, const uint8_t *data, size_t n) {
    return i2c1_write2(addr7, data, n, (const uint8_t *)0, 0);
}
