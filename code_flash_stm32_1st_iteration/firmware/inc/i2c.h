#ifndef I2C_H
#define I2C_H

#include <stddef.h>
#include <stdint.h>

/*
 * I2C1 on PB6 (SCL) / PB7 (SDA), fast mode 400 kHz, APB1 assumed 48 MHz.
 * Polled master-transmit only -- we never read from the OLED.
 */
void i2c1_init(void);

/* Send addr7 + data[0..n) with STOP. Returns 0 on success, <0 on timeout. */
int i2c1_write(uint8_t addr7, const uint8_t *data, size_t n);

/*
 * Same but sends `prefix` then `data` in one transaction. Convenient for
 * the SSD1306 which needs a leading 0x40 control byte before framebuffer
 * bytes, without allocating a 1 KB coalesced buffer.
 */
int i2c1_write2(uint8_t addr7,
                const uint8_t *prefix, size_t plen,
                const uint8_t *data, size_t dlen);

#endif /* I2C_H */
