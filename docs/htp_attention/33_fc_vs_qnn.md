<!-- SPDX-License-Identifier: Apache-2.0 -->

# 33 -- The fully connected layer against QNN, stage by stage

QNN reported a per-stage breakdown of one fully connected op rather than a
single number, which is what makes a real comparison possible: totals would
only say who is faster, and the stages say what each stack is spending its
time on. This records the mapping, the one number to compare when the question
is "compute only", and what QNN's numbers already settle before any measurement
of ours is in.

The report draws all of this from a live log:

    tools/htp_fc_report.py --qnn-profile

## 1. What QNN reported

One FC op, u8i4, reported on an S26 Ultra. Sequence length not stated -- §4
backs it out.

| stage | us | what it is |
|---|---|---|
| one-time init | 5,097 | context/binary load, once |
| cold run | 1,263 | first inference |
| **NetRun (steady state)** | **513** | execute() entry to return |
| . host / qnn-net-run overhead | 128 | NetRun - RPC execute |
| . backend RPC execute | 385 | host <-> device |
| .. non-accelerate RPC overhead | 39 | RPC execute - accelerate execute |
| .. **accelerate execute (device)** | **347** | the device timeline |
| ... input slice / load (uint8) | 44.5 | |
| ... **FC (weight load + compute)** | **69.9** | |
| ... output format / layout (uint8) | 151.1 | |
| ... output writeback | 72.9 | |
| ... idle / misc / overlap | 5.2 | |

## 2. The mapping

QNN's graph is **uint8 in and uint8 out**. Ours takes f32 and returns f32. So
there is no QNN stage that corresponds to our quantize or dequantize as such --
what their pipeline has instead is a layout/format stage that converts the
int32 accumulator to uint8, which is the same *work* our acc_read and dequant
do, at a quarter of the output bytes.

| QNN stage | our stage(s), from FC_STAGE |
|---|---|
| input slice / load (uint8) | `quant_us` |
| FC (weight load + compute) | `dsp_total - quant - dequant - acc_read - acc_copy` |
| output format / layout (uint8) | `acc_read_us + dequant_us` |
| output writeback | `acc_copy_us` (0 on the in-place path -- ours is folded into dequant's DDR store) |
| accelerate execute (device) | `dsp_total_us` |
| NetRun - accelerate execute (166) | `transport_us` |
| NetRun (steady state) | `us_avg_1_10` (wall) |
| one-time init | `init_us` (the weight bake) |

## 3. The answer: compute only is 69.9 us

**Compare against QNN's `FC (weight load + compute)` = 69.9 us**, and on our
side compute the same quantity by subtraction:

    compute = dsp_total_us - quant_us - dequant_us - acc_read_us - acc_copy_us

which is the micro-mms plus the weight DMA, and nothing else. That subtraction
is what the report's `--qnn-profile` section prints as `micro-mm + weight DMA`.

Do NOT compare against 347 or 513 when the question is compute: 347 carries a
uint8 layout stage we do not have and 513 carries a host stack we do not have.
Do not compare our `dsp_total` against 69.9 either -- that charges us for four
stages QNN accounts for separately.

## 4. What QNN's numbers settle on their own

**Their compute is 20% of their own device timeline.** 69.9 of 347. The other
80% is getting data into and out of the accumulator.

**Their output-format bucket is 2.2x their compute bucket.** 151.1 against
69.9 -- and 151 us cannot be data movement for an output this small. That is
an accumulator drain plus a requantize, which is exactly the finding our own
Tier 0 probes produced (31_dataflow_as_built.md §3: the readout, not the
multiply, was 75-96% of attention DSP time). **Both stacks are losing to the
same thing.** Any framing of this comparison as "their matmul is faster than
ours" misses that neither stack is spending its time on the matmul.

**The measurement is at M=1 (decode).** Their table does not say, and it
changes what 69.9 us means, so back it out of the number itself at q_proj's
shape (K 1024, N 2048):

| M | MACs | implied rate | verdict |
|---|---|---|---|
| 1 | 2.10 M | 30 GMAC/s | plausible |
| 512 | 1.07 G | 15,361 GMAC/s | above any HMX int8 peak |
| 1024 | 2.15 G | 30,722 GMAC/s | above any HMX int8 peak |

Only M=1 survives. It is corroborated from the other side: 1.0 MiB of i4
weight read in 69.9 us is **15.0 GB/s**, which is a DDR read rather than a
cache hit. So their FC op at decode is weight-load bound, 69.9 us is close to
what the memory system allows at this shape, and the room in their 513 us is
in the other 443.

## 5. What is not comparable, and must be said when quoting this

- **S26 Ultra vs S25 Ultra.** Different silicon. The attention table already
  carries this asymmetry; it applies here unchanged.
- **u8i4 vs u8i8.** Their weight is half the bytes, which matters precisely
  because §4 shows the op is weight-load bound at decode. Our i8 q_proj reads
  2 MiB where theirs reads 1. At the same bandwidth that is 2x on the one
  stage being compared, before any implementation difference.
- **f32 interface vs uint8 interface.** Ours quantizes on entry and
  dequantizes on exit, inside the measured call, and carries 4x the bytes over
  FastRPC in both directions. QNN does neither.

## 6. What follows from this

Open until the device run lands, stated as predictions so they can be wrong:

1. At M=1 our accumulator is 64 rows wide and one of them is useful. 2,048
   micro-mms at the measured 0.38 us each is ~778 us of issue for 2.1 MMAC of
   answer. If the run confirms that, the decode gap against 69.9 us is not
   bandwidth and not the kernel's arithmetic -- it is doing 64x the work, and
   the fix is a GEMV path that does not go through HMX at all (already on
   32_after_the_copy.md's list as 1c).
2. Prefill is the opposite regime and the comparison above does not describe
   it. At M=512/1024 the padding vanishes and the op becomes issue-bound; that
   is where the micro-mm cost model is the thing to argue with.
3. The f32 interface is a choice, not a constraint. `hexkl_mm_opts` already
   accepts caller-supplied `act_scale`/`act_zp`, so half of a u8-in endpoint
   exists. If the measured quant+dequant share approaches QNN's 56%
   (44.5 + 151.1 + 72.9 of 347), that endpoint is the next thing to build, and
   it would also cut the FastRPC payload by 4x.
