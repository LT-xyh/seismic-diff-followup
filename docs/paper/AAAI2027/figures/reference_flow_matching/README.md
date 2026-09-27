# Flow Matching Method-figure References

These cropped figures are collected from public arXiv papers for internal
figure-design reference only. They are not manuscript assets and should not be
reused in the submitted paper.

## References

1. `01_lipman_conditional_vector_field.png`
   - Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023.
   - arXiv: https://arxiv.org/abs/2210.02747
   - Figure 2, PDF page 6.
   - Useful pattern: show several time slices and make the target vector field
     visually explicit with arrows and color-coded magnitude.

2. `02_lipman_trajectory_comparison.png`
   - Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023.
   - arXiv: https://arxiv.org/abs/2210.02747
   - Figure 3, PDF page 6.
   - Useful pattern: compare curved diffusion trajectories with simpler OT
     trajectories in a small conceptual inset.

3. `03_lipman_checkerboard_transport.png`
   - Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023.
   - arXiv: https://arxiv.org/abs/2210.02747
   - Figure 4, PDF page 7.
   - Useful pattern: use a row of spatial states indexed by time, then add a
     compact sampling-efficiency comparison beside it.

4. `04_tong_conditional_flow_overview.png`
   - Tong et al., *Improving and Generalizing Flow-Based Generative Models with
     Minibatch Optimal Transport*, TMLR 2024.
   - arXiv: https://arxiv.org/abs/2302.00482
   - Figure 1, PDF page 3.
   - Useful pattern: distinguish conditional paths, marginal flow, and learned
     transport in one compact overview.

5. `05_liu_rectified_flow_interpolation.png`
   - Liu et al., *Flow Straight and Fast: Learning to Generate and Transfer Data
     with Rectified Flow*, ICLR 2023.
   - arXiv: https://arxiv.org/abs/2209.03003
   - Figure 2, PDF page 4.
   - Useful pattern: place the interpolation equation directly under each
     panel and use geometry to explain why the transport is simpler.

6. `06_latent_flow_matching_training.png`
   - Dao et al., *Flow Matching in Latent Space*, 2023.
   - arXiv: https://arxiv.org/abs/2307.08698
   - Figure 1, PDF page 4.
   - Useful pattern: combine data encoding, the probability path, velocity
     prediction, and decoding into a single training/sampling story.

7. `07_matcha_conditional_flow_architecture.png`
   - Melechovsky et al., *Matcha-TTS: A Fast TTS Architecture with Conditional
     Flow Matching*, 2023.
   - arXiv: https://arxiv.org/abs/2309.03199
   - Figures 1-2, PDF page 3.
   - Useful pattern: show the conditioning stream and the flow-prediction
     network separately, while still exposing the time-indexed states.

8. `08_sd3_conditioned_flow_architecture.png`
   - Esser et al., *Scaling Rectified Flow Transformers for High-Resolution
     Image Synthesis*, 2024.
   - arXiv: https://arxiv.org/abs/2403.03206
   - Figure 2, PDF page 5.
   - Useful pattern: AAAI/CVPR-style architecture drawing with explicit
     condition and timestep injection, typed blocks, and a small detailed
     module inset.

## Recommended synthesis for PD-BG-RFM Fig. 2

Use the conceptual language of references 1, 3, and 5, and the conditional
architecture language of references 6-8:

```
noise xi + structural condition C_S' + background context psi_B(B_hat)
                              |
                              v
             c_R = [C_S'; psi_B(B_hat)]
                              |
       R_t = (1-t) xi + t (V - B_hat),  t in [0, 1]
                              |
       v_theta(R_t, t, c_R) -> Euler transport -> R_hat
                              |
                   V_hat = B_hat + R_hat
```

For the paper figure, the most defensible visual evidence is:

- four or five time-indexed residual states (`t=0`, intermediate `t`, `t=1`);
- a small vector-field or velocity-arrow inset, following Lipman Figure 2;
- the explicit residual target and Euler update, following Rectified Flow;
- a distinct condition branch for `C_S'` and `psi_B(B_hat)`, following latent
  Flow Matching and Matcha-TTS;
- a dashed training-only branch for the target residual, with no target access
  on the inference path.

Avoid copying the original visual layouts or decorative elements. Recreate the
idea with the project's own seismic tiles, colors, and notation.
