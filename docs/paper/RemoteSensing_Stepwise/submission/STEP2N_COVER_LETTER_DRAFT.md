# Remote Sensing Cover Letter — Draft for Final Author Confirmation

**Status:** DRAFT ONLY — NOT APPROVED FOR SUBMISSION

**To:** Editors of *Remote Sensing* (MDPI)

**Re:** Article submission — *Predicting the Background, Transporting the Structure: Decoupled Multimodal Subsurface Velocity Inversion*

Dear Editors,

Please consider our manuscript, **“Predicting the Background, Transporting the Structure: Decoupled Multimodal Subsurface Velocity Inversion,”** as an Article in *Remote Sensing*.

Initial subsurface velocity model building relies on heterogeneous geophysical evidence with different reconstruction roles. RMS velocity and sparse well logs inform velocity trends and magnitudes, while post-stack time-migrated images and interpreted horizons constrain subsurface structure. These processed and interpreted observations motivate a reconstruction approach that coordinates broad background estimation and localized structural correction.

We present **Physics-Decoupled Background-Guided Residual Flow Matching (PD-BG-RFM)**, which learns role-aware multimodal conditions, directly estimates a smooth velocity background, and uses conditional Flow Matching to transport the residual relative to that same background prediction. The estimated background consistently defines residual supervision, conditions the learned vector field, and enters final composition. This organizes deterministic estimation and generative correction around an observation-dependent reference rather than proposing a new mathematical family of Flow Matching.

The manuscript reports matched-input results on **33,600 held-out records** from eight OpenFWI-derived velocity subsets. The proposed method achieves RMSE **0.0270**, SSIM **0.9913**, and high-frequency MAE **0.0065** with **10.6 million trainable parameters**, leading the evaluated adapted baselines on these measures. Controlled ablations, condition-routing measurements and transport-target diagnostics provide complementary evidence for this reconstruction organization under the documented synthetic processed-constraint protocol.

We believe this combination of multimodal geophysical conditioning and prediction-consistent generative reconstruction is relevant to the readers of *Remote Sensing* working on seismic interpretation, subsurface imaging and geophysical inverse problems.

**Required submission declarations — pending final confirmation by all three authors (not yet certified):**

- [ ] We confirm that neither the manuscript nor any parts of its content are currently under consideration for publication with or published in another journal.
- [ ] All authors have approved the manuscript and agree with its submission to *Remote Sensing*.
- [ ] We confirm that the manuscript presents original research and that the final manuscript, authorship and submission statements have been reviewed and approved.

The statements above must be affirmatively verified by Yinghao Xu, Chunlei Wu and Jing Lu before the letter is signed or uploaded. An unchecked draft is **not** a declaration of completed author approval.

Thank you for considering our manuscript.

Sincerely,

**Chunlei Wu**  
Corresponding Author  
College of Computer Science and Technology, China University of Petroleum (East China), Qingdao, China  
Shandong Province Key Laboratory of Intelligent Oil & Gas Industrial Software, Qingdao, China  
wuchunlei@upc.edu.cn

---

**Editorial control:** Draft prepared for author review only. Not submitted to MDPI. The manuscript title, author order, corresponding author and reported metrics have been updated from the historical draft to the approved Step 2M version. No related-manuscript discussion is inserted into this letter.
