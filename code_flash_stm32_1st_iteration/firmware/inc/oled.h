#ifndef OLED_H_
#define OLED_H_

#include <stddef.h>

/*
 * SSD1306 128x64 I2C OLED at 0x3C.
 * Depends on i2c1_init() having been (or being) called by oled_init().
 *
 * Text is drawn in a 6x8 grid: 21 columns wide, 8 rows tall (168 chars).
 * When the cursor overruns the bottom row, the framebuffer scrolls up.
 * Every putc triggers an incremental flush so text appears live.
 */

void oled_init(void);
void oled_clear(void);
void oled_putc(char c);
void oled_puts(const char *s);

#endif /* OLED_H_ */
