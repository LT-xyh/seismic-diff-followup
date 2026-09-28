# Draft Cover Letter — Remote Sensing

[DATE]

Editors  
*Remote Sensing* (MDPI)

Dear Editors,

Please consider our manuscript, **“Background-Guided Residual Flow Matching for Multi-Constraint Seismic Velocity Model Building,”** for publication as an Article in *Remote Sensing*.

Reliable initial velocity model building is important for seismic imaging and subsequent subsurface interpretation. In workflows where several processed and interpreted geophysical products are available, RMS velocity, migrated seismic information, interpreted horizons, and sparse wells provide complementary constraints but differ substantially in their spatial support and reconstruction roles. Our manuscript introduces Background-Guided Residual Flow Matching (BG-RFM), which explicitly assigns deterministic background recovery and generative residual transport different responsibilities.

The central methodological feature is prediction consistency. BG-RFM first estimates a deterministic background velocity and then models the prediction-relative correction using conditional Flow Matching. The same predicted background is reused to define the residual target, condition residual transport, and compose the final velocity model. This provides a structured multi-constraint reconstruction formulation rather than treating the entire velocity field as one undifferentiated generative target.

We evaluate the formulation across eight OpenFWI-derived subsets using a common multimodal observation contract. The unified matched-input evaluation contains 33,600 held-out records and compares BG-RFM with adapted deterministic and generative references under the same evaluator and normalization. BG-RFM achieves an RMSE of 0.0269 and an SSIM of 0.9916 with 10.6M trainable parameters. We do not claim overall benchmark dominance; the paper instead emphasizes the reconstruction formulation, prediction consistency, and compact parameterization, supported by controlled design analyses, qualitative examples, and diagnostic evidence.

We believe the manuscript is relevant to *Remote Sensing*, particularly its **Remote Sensing in Geology, Geomorphology and Hydrology** section, whose scope includes observation of Earth structure on and beneath the surface as well as exploration and oil-and-gas applications. The work combines multi-source geophysical information with modern generative modeling for subsurface structural reconstruction.

[AUTHOR CONFIRMATION REQUIRED: This manuscript has not been published previously and is not under consideration for publication elsewhere.]

[AUTHOR CONFIRMATION REQUIRED: All authors have approved the manuscript and agree with its submission to *Remote Sensing*.]

[IF APPLICABLE: Provide any prior MDPI manuscript ID and a brief explanation.]

Thank you for your consideration.

Sincerely,

[CORRESPONDING AUTHOR NAME]  
[AFFILIATION]  
[EMAIL]
