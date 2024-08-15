# ARM Neon Demo

For this demo, copy your code from the `loop_unrolling` folder.  You will use that as a starting point.

Your goal for this demo is use the ARM Neon Intrinsics to do a vector addition of 4 elements into one.  You will later leverage this for doing the ARM Neon Intrinsics.  Use the resources in the top-level readme as a guide.

1. Load the current 4 elements into a quadword
2. Sum all four elements of the quadword
3. Extract the value from the quad word
2. Increment by 4 instead of by 1

Questions to consider:

* How do we handle when there is not at least four elements remaining?
* What is happening at the hardware level that may make this more efficient?