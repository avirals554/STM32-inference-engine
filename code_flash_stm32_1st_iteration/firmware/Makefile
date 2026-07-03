# STM32F411CEU6 firmware build.
#
# `make`       -> build inference.elf + inference.bin
# `make size`  -> show flash/RAM footprint
# `make flash` -> write to chip via st-flash (STLink)

TARGET   := inference

CC       := arm-none-eabi-gcc
OBJCOPY  := arm-none-eabi-objcopy
SIZE     := arm-none-eabi-size

MCU      := -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard

CFLAGS   := $(MCU) -Os -std=c11 -g \
            -Wall -Wextra -Wno-unused-parameter -Wno-missing-braces \
            -ffunction-sections -fdata-sections \
            -Iinc -Igenerated

LDSCRIPT := stm32f411.ld
LDFLAGS  := $(MCU) -T$(LDSCRIPT) -nostartfiles \
            -Wl,--gc-sections -Wl,-Map=$(TARGET).map \
            --specs=nano.specs --specs=nosys.specs

LIBS     := -lm

SRCS     := src/startup.c src/uart.c src/i2c.c src/oled.c \
            src/main.c src/model.c src/tokenizer.c
OBJS     := $(SRCS:.c=.o)

all: $(TARGET).bin size

$(TARGET).elf: $(OBJS) $(LDSCRIPT)
	$(CC) $(OBJS) $(LDFLAGS) $(LIBS) -o $@

$(TARGET).bin: $(TARGET).elf
	$(OBJCOPY) -O binary $< $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

size: $(TARGET).elf
	$(SIZE) $(TARGET).elf

flash: $(TARGET).bin
	st-flash --reset write $(TARGET).bin 0x08000000

clean:
	rm -f $(OBJS) $(TARGET).elf $(TARGET).bin $(TARGET).map

.PHONY: all size flash clean
