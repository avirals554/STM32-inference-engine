#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>

#include "model_config.h"

/*
 * Runs the transformer over the given context window and returns logits for
 * the LAST position (index CONTEXT_LENGTH-1).  That is the only slot we ever
 * sample from for next-token prediction.
 *
 *   tokens : CONTEXT_LENGTH input token ids
 *   logits : output buffer of length VOCAB_SIZE
 *
 * The engine uses static work buffers in .bss (~96 KB); it is not
 * re-entrant.  Fine for STM32 single-thread use.
 */
void model_forward(const uint8_t tokens[CONTEXT_LENGTH],
                   float logits[VOCAB_SIZE]);

/*
 * Given VOCAB_SIZE logits, returns the argmax token id.  Greedy sampling.
 */
uint8_t model_argmax(const float logits[VOCAB_SIZE]);

#endif /* MODEL_H */
