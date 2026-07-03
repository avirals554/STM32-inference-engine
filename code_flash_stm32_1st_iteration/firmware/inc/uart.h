#ifndef UART_H
#define UART_H

#include <stddef.h>
#include <stdint.h>

/* USART1 on PA9/PA10, 115200 8N1, PCLK2 assumed 96 MHz. */
void uart_init(void);

void uart_putc(char c);
void uart_puts(const char *s);
void uart_write(const char *buf, size_t n);

/* Blocking read of one byte. */
uint8_t uart_getc(void);

#endif /* UART_H */
