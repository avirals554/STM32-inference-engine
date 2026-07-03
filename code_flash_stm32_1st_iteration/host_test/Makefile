# Host build of the portable inference engine for offline verification.
# Uses the SAME model.c/tokenizer.c as the STM32 firmware, so bugs surface
# here on the dev machine instead of on-chip.

CC       := cc
CFLAGS   := -O2 -std=c11 -Wall -Wextra -Wno-unused-parameter -Wno-missing-braces \
            -I../firmware/inc -I../firmware/generated
LDFLAGS  := -lm

SRC      := main.c ../firmware/src/model.c ../firmware/src/tokenizer.c
OBJ      := $(SRC:.c=.o)
BIN      := host_test

$(BIN): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(BIN) $(LDFLAGS)

run: $(BIN)
	./$(BIN) "Once upon a time"

clean:
	rm -f $(BIN)

.PHONY: run clean
