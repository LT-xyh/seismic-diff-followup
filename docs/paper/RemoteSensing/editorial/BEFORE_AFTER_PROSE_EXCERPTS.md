# Before/after academic prose examples

Frozen RS baseline: `cc71d310e06622b826ea5bd821e96f3a4e7e9ce6`.
The AAAI text is a literary/conceptual guide, not an authority for current science.

## Introduction: heterogeneous observation roles

**Before (compressed RS)**

> Data-driven velocity reconstruction provides a complementary route by amortizing the mapping from observations to a velocity model. The task studied here is specifically velocity-model reconstruction when processed or interpreted constraints are already available: RMS velocity, a post-stack migrated image, interpreted horizons, and sparse well velocities.

**After (reconstructed)**

> The observations available for this task differ fundamentally from the raw waveforms usually supplied to learned seismic inversion systems. RMS velocity provides a time-domain description of cumulative propagation kinematics and carries useful information about velocity scale. Post-stack migrated sections emphasize reflector configurations, while interpreted horizons express a more selective description of interface geometry. Well logs offer local depth-domain velocity control but cover only a small fraction of the lateral extent.

The revised paragraph establishes a geophysical distinction and then motivates the condition-design problem; it does not simply name the four modalities.

## Related Work: reconstruction variable rather than model catalog

**Before**

> Learning-based seismic inversion has developed from direct convolutional mappings between seismic measurements and subsurface velocity toward larger encoder–decoder, attention-based, and structured inverse architectures.

**After**

> The choice of observations is central to the inverse problem posed by a learning-based velocity model. Classical seismic velocity analysis and depth imaging derive their constraints from travel-time behavior and the propagation of wavefields, while full-waveform inversion seeks a velocity field whose simulated measurements agree with observed data.

The argument now differentiates observation contracts, training criteria, estimator types and transported variables, while retaining the original citations' scope.

## Methods: inference-available residual reference

**Before**

> The residual target is defined relative to the background that is actually available at inference.

**After**

> A residual generator requires a reference field, but an oracle low-pass target is unavailable for an unseen example. BG-RFM therefore defines the correction relative to the predicted background that will also be used at inference.

The revised prose supplies a reason for the unchanged equation rather than repeating the variable's definition.

## Results: benchmark evidence in context

**Before**

> BG-RFM obtains an RMSE of 0.02695 and an SSIM of 0.99165 with 10.63M trainable parameters. Relative to the InversionNet adaptation, BG-RFM has lower RMSE ... [then a sequential metric inventory].

**After**

> The Latent U-Net adaptation achieves the strongest aggregate reconstruction accuracy, with MAE 0.00706, RMSE 0.01309, and SSIM 0.99403. It is therefore the most accurate estimator among those listed in Table 2. BG-RFM gives MAE 0.01503, RMSE 0.02695, and SSIM 0.99165 with 10.63M trainable inference parameters, compared with 35.12M for the adapted Latent U-Net.

This leads with the numerically strongest method and interprets the proposed model as a different, compact generative organization.

## Discussion: substantive academic argument

**Before**

> Why a predicted-background reference matters. BG-RFM assigns heterogeneous observations two related but distinct reconstruction roles ...

**After**

> The reconstruction problem considered here is shaped by an unusual combination of information sources. RMS velocity provides a broad kinematic description, migrated imagery and horizons emphasize structure, and sparse wells constrain velocity locally in depth. Encoding these observations jointly is necessary, but their fusion does not prescribe how a velocity field should be recovered from the resulting representation.

Question-style report headings become continuous geophysical reasoning. Subsequent paragraphs discuss the full-field transport equivalence, benchmark tradeoffs, diagnostics and synthetic observation scope.

## Conclusion: methodological synthesis

**Before**

> We introduced Background-Guided Residual Flow Matching (BG-RFM) for seismic velocity reconstruction from heterogeneous processed and interpreted constraints.

**After**

> Initial velocity reconstruction from heterogeneous geophysical constraints is not only a question of how to fuse observations but also of how to assign the information they contain to different parts of a reconstruction.

The conclusion begins from the resolved scientific design problem rather than from a second method inventory.
