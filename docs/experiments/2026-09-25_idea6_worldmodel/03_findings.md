# Findings

1. **Stage A works, modestly.** Both world models beat persistence by roughly 55-59% at 8 steps. HN edges MLP (0.132 vs 0.144
   at h=8), but it has 2x the parameters and one seed, so the Hamiltonian structure cannot be credited. Much of the skill is
   probably learning the systematic drift (vitals decay) that persistence lacks; a simple linear-drift baseline was not run and
   could close the gap.
2. **Stage B: no benefit from the world model.** The four arms are indistinguishable with n=2 seeds. The untrained-HN control
   (3.14) is not worse than trained HN (2.94), so the extra features carry no usable signal for this policy at this budget.
3. **The large jump over random is from the adapter fix.** Before, nothing beat random (return <= 1.4 vs 1.46); with the
   observation exposed, all arms reach about 3 (crafter about 3.5). Earlier "architecture" comparisons on the old adapter measured a
   blind policy; those candidate conclusions should be re-run on the new adapter (only transformer_branch was here).
4. **Why the WM cannot obviously help here (design critique).** The predicted quantity (passive vitals drift) is nearly a
   function of vitals the policy already sees, and the decisions that matter in Craftax (where to go, what to craft) are not about
   passive vitals. A model of *action* effects on map/inventory would be the informative one, but that is the part the "do
   nothing first" spec defers, and it would need simulator-like knowledge.
5. **Not tested / cannot claim:** "LLM-level results at lower cost". No LLM or published-agent baseline was run, and 4,800
   episodes of REINFORCE is not comparable to published Craftax numbers. Event jumps, whether the 2 gated modes mean anything,
   and the Monte-Carlo module were not exercised.
6. **Decisive follow-ups:** at least 5 seeds and 3x longer runs for base vs wm_hn vs untrained; a linear-drift baseline for
   Stage A; give the WM action inputs (u) and use it for planning rather than as static features.
