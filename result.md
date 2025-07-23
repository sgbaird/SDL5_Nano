# **A Data-Driven Workflow for Nanomedicine Optimization Using Active Learning and Automated Experimentation**

**Authors**

Zeqing Bao1\*, Frantz Le Devedec1, Steven Huynh2

**Affiliations**

1. Acceleration Consortium, University of Toronto, ON M5S 3H6, Canada
2. Leslie Dan Faculty of Pharmacy, University of Toronto, Toronto, ON M5S 3M2, Canada

\*Corresponding author: Zeqing Bao (zeqing.bao@utoronto.ca)

# **Abstract**

Nanomedicines are an advanced class of drug formulations that hold significant promise, particularly in enhancing the solubility of hydrophobic drugs. However, current state-of-the-art methodologies for developing nanomedicines are often inefficient, limiting both the systematic screening of dosage forms and the fine-tuning of individual formulations. To overcome these challenges, this study introduces a data-driven workflow that integrates active learning with experimental automation to rapidly identify optimal nanoformulations, using aceclofenac as a model poorly soluble drug. The initial formulation design space comprised combinations of the drug with 12 different excipients, resulting in approximately 17 billion possible formulations. To optimize across four objectives simultaneously, the active learning–robotic system efficiently narrowed this vast space to a manageable subset. This refined subset was further explored using a design of experiments approach, with selected formulations manually prepared and subjected to standardized evaluation. Within weeks, a panel of high-performing lead nanoformulations was identified. Notably, several of these promising formulations represent hybrid nanomedicines that are not well studied in the literature. These findings highlight the power of combining AI-driven design with automation to accelerate nanomedicine development and lay the groundwork for more efficient formulation development.

![](data:image/x-emf;base64...)

**Graphical Abstract**

Keywords: Nanomedicine, Active Learning, Bayesian Optimization, Automated Experimentation, Self-Driving Lab

# **Introduction**

Nanomedicines represent one of the most significant advancements in modern drug delivery, with over 50 nano-drugs approved by the U.S. Food and Drug Administration (FDA) to date [1,2]. Nanomedicines include a broad spectrum of dosage forms, including polymer nanoparticles, lipid nanoparticles, micelles, nanocrystals, etc. as well as other hybrid systems [3–5]. Each formulation type utilizes various excipients or excipient combinations, offering unique advantages and enabling compatibility with a wide range of drug compounds. A key application of nanomedicines is enhancing the apparent solubility of hydrophobic drugs in aqueous media, which facilitates administration and improves therapeutic performance [6,7]. This is especially important for drugs classified under the Biopharmaceutics Classification System (BCS) as Class II, where low aqueous solubility significantly limits their oral bioavailability [8].

Traditionally, the development of drug formulations has relied heavily on trial-and-error methods. This approach is widely recognized as inefficient, particularly when navigating the vast and complex design space associated with nanomedicine [9,10]. Its time-consuming and costly nature restricts the depth and breadth of formulation optimization given the finite resource constraints [11]. As a result, formulations may enter clinical trials without being fully optimized. Moreover, these limitations often make it impractical to evaluate multiple dosage forms within a single optimization process, potentially overlooking superior formulation options.

In recent years, machine learning (ML) has emerged as a powerful data-driven tool to accelerate pharmaceutical formulation development [12–15]. For example, ML has been used to efficiently explore the design space of lipid-based nanoparticles [16]. By screening over 1,000 formulations in silico, the authors identified two lead candidates with superior performance within just a few weeks [16]. While that study [16] and others [17–23] demonstrate the potential of ML in formulation science, most of them rely on supervised ML models. Typically, these supervised ML models are usually trained on labeled datasets to predict formulation performance based on input variables such as excipient composition and processing techniques [24–26]. Once trained, they can generate in silico predictions that significantly reduce experimental workload. However, for these models to generalize effectively across large and diverse formulation spaces, access to extensive, high-quality datasets is essential [27]. In the context of nanomedicine development, such datasets are currently scarce and generating them through in-house experimentation remains costly and time-intensive. Although literature mining offers an alternative data collection method, inconsistencies in synthesis and characterization protocols between research groups pose challenges. These variations can significantly affect formulation performance, yet are frequently underreported or difficult to encode as model features, ultimately limiting predictive accuracy [28]. To address these challenges, active learning has emerged as a promising alternative within the ML framework. Unlike traditional supervised ML, which focuses on prediction, active learning operates as an iterative planning strategy [29,29–31]. It starts with little or no data and selects the most informative experiments at each step based on previous results [32]. This makes it well-suited to pharmaceutical formulation, where experimental resources are usually limited and the design space is large [33,34]. Furthermore, active learning can also be integrated with robotic systems for experimental execution, forming semi- or fully self-driving laboratories (SDLs) [35,36]. These systems have already shown great promise in accelerating discovery in materials science and chemistry [37–39].

In this study, we present a novel workflow that integrates active learning with robotic automation to optimize nanomedicine formulations across a wide range of excipients and dosage forms (**Figure 1**). To demonstrate the effectiveness of this approach, we selected aceclofenac (ACF, aqueous solubility 0.06 mg/mL) as a model compound to represent BCS Class II drugs with poor aqueous solubility. Nanoformulations of ACF were prepared using an automated liquid-handling system, with the formulation space explored iteratively using an active learning algorithm. Once the design space was narrowed to a manageable size, further refinement was conducted using a design of experiments (DOE) approach and benchtop manual preparation to identify lead candidates. This workflow enabled rapid exploration and optimization of the formulation space, identifying high-performing nanoformulations within weeks. Notably, several of the top candidates included hybrid nanomedicines that are not well-studied in the literature. These results highlight the effectiveness of the proposed workflow in enabling comprehensive and efficient nanomedicine development.

![](data:image/x-emf;base64...)

**Figure 1**. Overview of the study workflow. a) Preparation of the aqueous and organic phases, followed by their mixing to form different types of nanoformulations depending on the excipients used. b) Initial high-throughput screening using automated nanoformulation synthesis and characterization to explore a broad design space. This step is guided by active learning to iteratively improve formulations. c) A refined optimization process based on a narrowed design space from (b). This stage employs design of experiments and manual formulation preparation/characterization to identify the most promising nanoformulation candidates.

# **Results**

## **2.1. Formulation design space**

Various types of excipients were included in the initial formulation design, including polymers, solid lipids, liquid lipids, and aqueous surfactants. The goal of this broad excipient inclusion of was to allow the optimization across a panel of nanoformulations, including solid lipid nanoparticles, nanostructured lipid carriers, drug crystals, micelles, polymer nanoparticles, and other hybrid nanoformulations. In total, 12 excipients were selected (as shown in **Table 1**) based on their common usage in nanoformulations and their potential for enhancing the delivery of hydrophobic drugs [41–44].

Since a continuous design space theoretically allows for infinite combinations, the component ratios were discretized into 5% intervals to make the space computationally tractable. The number of possible combinations for the organic and aqueous phases was then estimated using the stars and bars combinatorial method, as described in the Methods section. Specifically, the organic phase, consisting of one drug and nine excipients, resulted in approximately 10 million possible combinations. The aqueous phase, comprising three surfactants and water for dilution, contributed an additional 1,700 combinations. When combined, the total estimated formulation design space reached around 17 billion possible formulations. This enormous space is clearly impractical to explore through conventional trial-and-error approaches. To address this, an active learning and automation workflow was developed to efficiently accelerate formulation screening and optimization.

|  |  |  |
| --- | --- | --- |
| **Table 1**. A list of excipients in the initial design space, with their abbreviations, and associated organic or aqueous phases used in nanoprecipitation. | | |
| Excipient | Abbreviation | Phase |
| PEG-PLGA (2k-2k) | P\_1 | Organic |
| PEG-PLGA (2k-5k) | P\_2 | Organic |
| PEG-PLGA (5k-5k) | P\_3 | Organic |
| Gelucire 50/13 | SL\_1 | Organic |
| Gelot 64 | SL\_2 | Organic |
| Stearic acid | SL\_3 | Organic |
| 2-(2-Ethoxyethoxy)ethanol | LL\_1 | Organic |
| Plurol® Oleique CC 497 | LL\_2 | Organic |
| PeceolTM | LL\_3 | Organic |
| Tween 80 | S\_1 | Aqueous |
| PluronicTM F-127 | S\_2 | Aqueous |
| Polyvinyl acetate | S\_3 | Aqueous |

## **2.2. Active learning-automation driven fast screening materials**

The combination of active learning and robotic systems is often referred to as semi- or fully self-driving lab workflows. In such workflows, active learning serves as a planning tool to determine the most informative experiments needed to achieve target objectives, while robotic systems execute those experiments.

In this study, active learning and automation were integrated to efficiently navigate the vast initial design space and identify a promising subset that met four key objectives: 1) particle size, 2) PDI, 3) theoretical drug loading, and 4) formulation complexity. The optimization targets for particle size and PDI were minimization. For particle size, although there is no clear consensus on the ideal particle size for enhancing oral bioavailability, current literature generally favors smaller particles for improved absorption [45–47]. In addition, a lower PDI indicates a more uniform particle size distribution, which is desirable for formulation consistency and performance. Theoretical drug loading was defined as the product of drug input and formulation quality. Formulation quality was assessed using the DLS plate reader, where a value of 1 indicated an acceptable measurement quality, and 0 signified poor quality. By maximizing the product of drug input and formulation quality, the algorithm was encouraged to incorporate higher drug concentrations while penalizing excessive drug content that led to precipitation. The final optimization objective, formulation complexity, was defined as the number of excipients used. Complexity was minimized to ensure practical formulation considerations and eliminate non-contributing excipients as optimization progressed.

To improve data generation efficiency, an automated protocol was developed for nanoformulation preparation in a 96-well plate format using a liquid handling robot. This protocol was based on our previous study [16], with detailed implementation available on GitHub (https://github.com/ZeqingBao/SDL5\_Nano). Using this automated approach, an initial random iteration (iteration 0) and 10 Bayesian Optimization iterations (iterations 1–10) were performed, as summarized in **Figure 2**. Each iteration contained 48 samples (16 unique formulations in triplicate), resulting in the screening of over 500 formulation experiments in total. As shown in the figures, all four optimization targets improved progressively. For example, particle size was reduced from an initial 250–300 nm to approximately 150 nm, PDI decreased from 0.35–0.4 to 0.2–0.3, theoretical drug loading increased from 0.1 to approximately 0.4, and formulation complexity decreased from 12 excipients to around 4.

**![A graph of growth and progress

AI-generated content may be incorrect.](data:image/png;base64...)**

**Figure 2.** Boxplots and best-so-far trace plots illustrating the progression of the four optimization targets over iterative design cycles: (a, b) size, (c, d) PDI, (e, f) theoretical loading, and (g, h) formulation complexity. Iteration 0 represents the initial random sampling (Sobol) used to seed the optimizer, while the subsequent ten iterations were guided by Bayesian optimization. Each iteration includes 16 unique formulations (n = 3 per formulation), shown as individual circles in the boxplots, with medians and interquartile ranges depicted. The best-so-far trace plots (right panels) track the optimal value identified up to each iteration, revealing the cumulative progress of the optimizer. Together, these plots demonstrate the effectiveness of active learning in steering the optimization process: minimizing size, PDI, and complexity, while maximizing theoretical loading. For clarity, only analyzable formulations are shown for boxplots; complete data are provided in Figure S1

Formulation optimization is inherently a multi-objective problem, evaluating overall performance across all objectives is therefore more meaningful than assessing each objective in isolation. In this study, hypervolume was used as a metric to quantify the overall effectiveness of the formulations and the optimization power of the algorithm. As shown in **Figure 3a**, the hypervolume of each formulation was calculated (represented by circles) and grouped by optimization iteration (represented by box plots) to illustrate the optimization trend. The resulting plots demonstrate a steady increase in hypervolume across successive iterations, indicating continuous improvement in formulation performance driven by the active learning process. Using hypervolume as a selection criterion, the top ten formulations were identified, and their excipient compositions are presented in **Figure 3b**. The results revealed that most top-performing formulations primarily relied on P\_3 (PEG-PLGA, 5k-5k) as the main organic component, often supplemented with lipids: Plurol® Oleique CC 497 or PeceolTM. For the aqueous phase, all top formulations consistently favored S\_2 (P407) as the surfactant. These findings provide valuable insights for guiding the next stage of optimization.

![A graph and chart of a graph

AI-generated content may be incorrect.](data:image/png;base64...)

**Figure 3.** (a) Hypervolume of individual nanoformulations (circles) grouped by optimization iteration (boxplots). Boxplots summarize the distribution of hypervolumes at each iteration, showing the median, interquartile range, and outliers. Iteration 0 corresponds to the initial random sampling used to seed the optimizer, while the subsequent ten iterations were guided by Bayesian optimization. The increasing trend in hypervolume indicates improvements in solution quality over successive iterations. For clarity, only analyzable formulations are shown; complete data are provided in Figure S2. (b) Heatmap visualization of the top ten identified formulations based on the hypervolume metric. Each row represents a formulation, with color intensity indicating excipient composition levels, darker shades correspond to higher concentrations. This visualization highlights the excipient preferences among the top-performing formulations.

## **2.3. DOE-manual formulation screening**

Following the active learning-automation workflow, four promising excipients were identified from the initial design space. The next step involved a refined manual investigation to identify optimal formulations. To determine the most relevant input ratios for each excipient, the distribution of excipients among the top ten formulations identified by the active learning process was visualized using boxplots (**Figure 4a**). In general, the PEG-PLGA (5k-5k) served as the primary component of the organic phase, with input ratios typically ranging from 0.3 to 0.5 (w/w), optionally supplemented by lipids (Plurol® Oleique CC 497 or PeceolTM) up to 0.2 (w/w). In the aqueous phase, surfactant (P407) concentrations commonly fell between 0.3% and 0.7% (w/v).

Based on these observed trends, a four-factor (excipient types), three-level (excipient ratios) experimental design was constructed, as summarized in **Figure 4b**. The pyDOE2 Python package was then used to process this design space and facilitate the systematic selection of representative formulations. While a full grid search for a 4-factor, 3-level design would result in 81 formulations, the DOE approach efficiently reduced the number of required formulations to 20. This reduction maintained a well-balanced distribution of factors and levels, significantly minimizing manual experimental workload while preserving meaningful design coverage.

![](data:image/x-emf;base64...)

**Figure 4.** (a) Excipient distribution of the top ten formulations identified through the active learning-automation optimization process, visualized using boxplots. The boxplots display the variability in excipient composition across the selected formulations, showing the median values and interquartile ranges. (b) Based on this distribution, a four-factor, three-level experimental design was constructed to systematically explore the refined formulation space. The experimental design ensures balanced coverage of excipient compositions for further validation and optimization. The four factors, LL\_2, LL\_3, P\_3, and S\_2, are abbreviations for Plurol® Oleique CC 497, Peceol™, PEG-PLGA (5k-5k), and P407, respectively.

A commonly used manual nanoprecipitation method was employed to prepare and analyze the formulations in the refined design space. This benchtop method enabled a tenfold increase in sample volume to facilitate formulation characterization and also allowed the use of membrane filtration to remove unformulated drug and excipients for drug quantification. Using this approach, the 20 formulations selected through DOE were prepared in triplicate. Each was evaluated for particle size, PDI, drug concentration, and stability. As shown in **Figure 5**, these formulations exhibited a range of characteristics: particle sizes varied from 40 to 160 nm, PDI values ranged from 0.1 to 0.4, and drug concentrations spanned from 200 to 1000 µg/mL, with differing levels of stability across samples.

Depending on the drug properties and intended use, different optimization campaigns may prioritize different formulation characteristics. To facilitate the selection of an optimal formulation based on specific requirements, a Pareto front was constructed to quantify the trade-offs between performance metrics among the candidate formulations. As shown in **Figure 6**, the three subplots illustrate the relationships among particle size, PDI, and drug concentration, with stability represented by marker shape. Therefore, formulations on the Pareto front can be selected as the promising candidates based on specific performance priorities. For example, if both stability and concentration are key priorities, Formulation 14 on the Pareto front is a suitable candidate. This formulation remained stable for two weeks, exhibited high drug solubility (~1 mg/mL), and maintained a small particle size (~100 nm) and low PDI (0.18).

![A screenshot of a diagram

AI-generated content may be incorrect.](data:image/png;base64...)

**Figure 5.** Results from the 20 formulations selected by DOE and manually prepared, visualized using boxplots. Each row of subplots represents a distinct property of interest: particle size, polydispersity index (PDI), drug concentration, and stability. Each column shows the effect of a specific excipient content on the corresponding property, highlighting how formulation composition influences key performance metrics. Individual formulations (n = 3) are shown as circles, with boxplots indicating medians and interquartile ranges.

![](data:image/x-emf;base64...)

**Figure 6**. Trade-off plots and summary table of the 20 formulations selected by DOE and manually prepared. (a) Size vs. PDI, (b) Size vs. Concentration, and (c) PDI vs. Concentration illustrate the trade-offs between different performance metrics. The density contour maps highlight the distribution of formulations, with Pareto-optimal solutions shown in red and non-Pareto solutions in blue. Stability is indicated by crosses (stable) and circles (unstable). (d) A summary table presenting the formulation parameters and results of these formulations.

# **Discussion**

## **3.1. Use of DOE and active learning for efficient design space exploration**

DOE and active learning are both powerful strategies for efficiently exploring design spaces. While both aim to minimize experimental workload, they are based on different theoretical foundations and offer different advantages depending on the research context. DOE originates from statistical and engineering disciplines and is designed to systematically plan experiments across the design space [48]. By strategically distributing experimental conditions, DOE allows researchers to identify a reduced yet representative set of data points that capture the underlying structure of the entire space [49,50]. DOE approach has been widely adopted across scientific fields and well-recognized in nanomedicine research, including the development of polymer nanoparticles [51,52], lipid nanoparticles [53,54], nanocrystals [55,56], etc. In contrast, active learning is a more recent methodology rooted in ML. Rather than uniformly sampling the design space in advance, active learning operates as a sequential optimization strategy that iteratively selects the most informative experiments based on the current models and available data [29,57,58]. This approach is particularly valuable when the design space is vast and experimental evaluations are costly, as it prioritizes experiments expected to yield the greatest improvement in model performance or progress toward predefined optimization objectives.

In this study, we strategically integrated both methods at different stages of the optimization process to leverage their respective strengths. Given that the initial formulation space was estimated to contain 17 billion possible formulations, DOE was insufficient to generate a representative yet manageable experimental plan. Active learning was therefore employed to guide experiment selection toward formulations that met predefined optimization targets. As shown in **Figures 2** and **3**, progressive improvements were observed across multiple objectives and in overall formulation performance as the optimization process advanced. This highlights the value of active learning–driven exploration, particularly in multi-objective optimization scenarios where human intuition may be insufficient, and objectives may conflict or interact in complex ways. Once the initial screening phase had narrowed the formulation space, DOE was employed in the subsequent optimization phase. At this stage, DOE offered two major advantages. First, as shown in **Figure 5**, the selected factors and levels were systematically distributed, resulting in a balanced dataset well-suited for downstream data analysis and/or for training supervised machine learning models when necessary. Second, unlike active learning, DOE enables all planned experiments to be executed simultaneously without requiring intermediate analysis. This is especially beneficial for time-intensive characterizations such as stability testing, where all formulations can be prepared and incubated in parallel, and measurements taken after a fixed period (e.g., two weeks). In contrast, active learning typically requires each iteration to be completed and analyzed before selecting the next batch of experiments, significantly extending the experimental timeline. By combining active learning and DOE in a complementary fashion, our approach not only enhanced experimental efficiency but also enabled more effective exploration and optimization of a complex, high-dimensional formulation space.

## **3.2. Use of automation and manual preparation in nanomedicine development**

Historically, drug formulations have been developed manually by formulation scientists [59,60]. More recently, the rise of automated experimentation has increasingly influenced the field of nanomedicine development, enabling more efficient and systematic approaches for formulation screening and optimization [61,62]. Automation offers several key advantages over traditional manual preparation, including high-throughput capabilities, streamlined workflows, and reduced human intervention. Additionally, these platforms support miniaturized, parallel experimentation, which reduces material consumption and overall costs.

Pioneering studies have demonstrated the potential of automated systems in the development of various nanomedicines. To date, much of these studies (including this work) has focused on nanoprecipitation [16,63–71], a nanomedicine formulation technique introduced in 1989 [72]. Nanoprecipitation has garnered attention in automation due to its simple setup [73–75], which is compatible with basic robotic systems such as liquid-handling robots. Moreover, it can be used to formulate a range of nanomedicines, including polymer nanoparticles [76,77], lipid nanoparticles [78,79], nanocrystals [80,81], and hybrid nanoparticles [82,83]. In this study, deploying an automated nanoprecipitation platform significantly increased experimental throughput. While manual preparation allowed for only 5 to 10 formulations per day, the automated system enabled the preparation and analysis of 48 formulations within just two hours, representing a 20- to 40-fold improvement. Furthermore, excipient consumption was reduced by approximately 90% due to the use of miniaturized sample volumes.

Despite these advantages, automation still faces key limitations compared to manual approaches, particularly in the context of nanomedicine screening. One major bottleneck is the lack of efficient, automated techniques for purification and characterization of nanoscale drug formulations. Traditional separation methods (e.g., centrifugation and membrane filtration) are still challenging to integrate into high-throughput workflows due to volume constraints and limited compatibility with existing equipment and labware. As a result, these steps often require manual intervention, which slows the overall process. To mitigate this challenge, the first phase of our automated screening was designed to map a promising formulation design space using a composite evaluation metric: the product of drug input ratio and formulation quality. Maximization of this metric encouraged the algorithm to explore formulations with higher drug loading while penalizing those with excessive drug content that led to precipitation. Following this rapid, automated screen, manual techniques were used for in-depth formulation preparation and characterization. The standard benchtop nanoprecipitation and analytical methods were employed to thoroughly characterize lead formulations, ensuring the robustness and reproducibility of the identified nanomedicine candidates.

## **3.3. Optimal formulations identified through the proposed data-driven workflow**

Multi-objective optimization is a common challenge in formulation science, as an ideal formulation is expected to perform well across multiple, often conflicting, criteria [84,85]. Because of these inherent trade-offs, there is rarely a single “best” formulation. Instead, a set of optimal candidates need to be carefully evaluated based on specific priorities. To address this, a Pareto front was used in this study to represent a set of non-dominated solutions, meaning that no formulation in this set can be improved in one objective without compromising another [86]. This approach offers an understanding of the trade-offs between competing objectives, enabling informed, priority-based decision-making. This structured selection process is particularly useful in guiding human decisions, as different projects may prioritize different performance metrics.

Using the proposed data-driven workflow, a set of promising Pareto-optimal nanoformulations was identified from an initial design space of approximately 17 billion formulations. This vast search space included numerous combinations of drug-to-excipient ratios and various excipient selections, allowing for formulation across multiple dosage forms. Such comprehensive exploration enables the identification of formulations that are not well-established in existing domain knowledge. For example, when comparing Formulation 3 and Formulation 14, both use the same drug input (50%) and surfactant concentration (0.75%). However, Formulation 3 relies solely on polymer (50%), whereas Formulation 14 includes both polymer and liquid lipid (30% polymer + 20% lipid). This seemingly minor change resulted in significant differences in performance. While both formulations achieved similar drug concentrations (~1 mg/mL), Formulation 3 produced nanoparticles around 30 nm in size but precipitated after two weeks. In contrast, Formulation 14 formed larger particles (~100 nm) but remained stable over time (Figure S3).

Specifically, Formulation 3 represents a classic polymer micelle system that takes advantage of the amphiphilic structure of PEG-PLGA. These micelles are widely reported in the literature for their ability to enhance the solubility of hydrophobic drugs [87–89]. They are also known for forming small nanocarriers, typically with diameters under 30 nm [90], which aligns with our observations (**Figure 6**). However, while incorporating hydrophobic drugs into micelles can enhance drug solubility, it may also increase the risk of micelle destabilization as observed in Formulation 3. In contrast, substituting 20% of the polymer content with Plurol® Oleique CC 497 in Formulation 14 resulted in improved stability over the course of the study, while still maintaining a high drug concentration. Plurol® Oleique CC 497, chemically known as polyglyceryl-3 dioleate, is a liquid lipid commonly used in self-nanoemulsifying drug delivery systems (SNEEDS) as a co-solubilizer and co-surfactant to enhance drug solubility and formulation stability [91–93]. In this case, the addition of the lipid likely introduced more oil-phase components, which disrupted the tight packing of the micelle core. This expansion may have led to a looser micelle structure and increased particle size, while ultimately contributing to micelle stabilization.

Importantly, although hybrid nanocarriers like Formulation 14 have shown strong performance, they remain underexplored in the literature. This is partly because such systems typically involve a vast design space, which traditional development methods struggle to investigate comprehensively. By leveraging active learning, automation, and DOE, the proposed workflow effectively navigated this complexity and identify formulations that extend beyond what is typically captured by existing literature.

## **3.4. Limitations**

While this study marks a critical step toward the development of a SDL for nanomedicine development, it is important to acknowledge its limitations. The drug ACF, used as a model compound, is currently administered in a 100 mg tablet form. In contrast, the optimized nanoformulation developed here achieves an aqueous solubility of approximately 1 mg/mL. This means that 100 mL of solution would be required for oral administration, which may pose practical challenges for drug delivery. Nonetheless, ACF was intentionally selected to represent BCS Class II compounds due to its affordability and accessibility, making it well-suited for demonstrating the proposed workflow. Depending on the specific application or characterization requirements, future studies may further enhance drug solubility by preparing more concentrated stock solutions or applying downstream concentration techniques such as tangential flow filtration or lyophilization.

# **Conclusion**

This study presents an efficient workflow that integrates active learning, automation, and DOE to strategically explore and optimize nanoformulations for hydrophobic compounds. Starting from an initial design space of 17 billion possible formulations, the combined use of active learning and automation enabled rapid narrowing to a manageable subset. This refined subset was then evaluated using DOE, followed by manual benchtop preparation to ensure standardized purification and analysis. Using this workflow, several lead candidates were identified within weeks from the initial vast design space. Notably, the process enabled the discovery of stable hybrid formulations that might have been overlooked using traditional approaches. Overall, this study highlights the effectiveness of combining automation and active learning in nanomedicine development, representing a key step toward SDLs for advanced drug formulation.

# **Materials and Methods**

## **5.1. Materials**

Stearic acid (SA, 95%), CarbitolTM (DGME, 2-(2-Ethoxyethoxy)ethanol, 99%), tetrahydrofuran (THF, HPLC grade), PluronicTM F-127 (P407), and poly(vinyl alcohol) (PVA, 13k-23k) were purchased from Sigma Aldrich (ON, CA). Three methoxy poly(ethylene glycol)-b-poly(lactide-co-glycolide) (PEG-PLGA; 2k-2k, 2k-5k, and 5k-5k) with 50:50 LA:GA ratios polymers were purchased from PolySciTechTM (IN, US). Gelucire 50/13®, GelotTM 64, Plurol® Oleique CC 497, and PeceolTM were gifts from Gattefossé Canada Inc (ON, CA). Aceclofenac (ACF, 98%) was purchased from Fisher Scientific (MA, US). Ethanol (anhydrous) was purchased from Greenfield Global (ON, CA). Tween 80® (Polysorbate 80, reagent grade) was purchased from BioShop (ON, CA). Methanol (HPLC grade) was purchased from Caledon Laboratory Chemicals (ON, CA).

## **5.2. Active learning-automation formulation screening**

### 5.2.1. Design space

The objective of this study was to develop nanoformulations for a hydrophobic compound to enhance its performance. ACF was selected as the model compound due to its poor aqueous solubility and FDA approval. The design space consisted of 12 excipients, including pegylated polymers, solid lipids, liquid lipids, and aqueous surfactants, as listed in **Table 1**. To estimate the size of the design space, the stars and bars combinatorial method was used. Each formulation phase (organic or aqueous) consisted of a combination of the drug and excipients, with the total component ratio normalized to 1. The ratios were discretized in 5% intervals, yielding 20 units per phase. The number of possible combinations for a given phase was calculated using the combinatorial formula, where n is the number of units (i.e., 1 divided by the smallest ratio step, which is 0.05, resulting in 20), k is the number of components (drug and excipients) in the aqueous or organic phase:

### 5.2.2. Automated nanoformulation preparation

An automated protocol was adapted from our previously published study to enable high-throughput nanoformulation preparation [16]. This protocol was implemented on a liquid handling robot (OT-2, Opentrons, NY, US), with the automation protocol available on GitHub (https://github.com/ZeqingBao/SDL5\_Nano). This protocol utilized the nanoprecipitation technique, wherein separately prepared organic and aqueous phases were mixed to form nanosized formulations. Prior to automation, stock solutions of the drug and excipients were manually prepared and loaded into 20 mL scintillation vials. These stock solutions were grouped into two categories: the organic phase and the aqueous phase. Specifically, the organic phase contained the drug, polymers, and solid/liquid lipids, all dissolved in THF at a concentration of 20 mg/mL. The aqueous phase consisted of the three surfactants dissolved in water at 1% (w/v), along with one blank vial of pure water for surfactant dilution. To enhance efficiency and reduce material consumption, each nanoformulation was prepared in a total volume of 300 µL using a 96-well plate format. The OT-2 robot first used a single-channel pipette to prepare the organic and aqueous working solutions according to predefined ratios. Then, using an 8-channel pipette, 30 µL of the organic phase was injected into 270 µL of the aqueous phase. This was followed by 20 cycles of mixing cycles (aspiration and dispensing) to induce nanoprecipitation and generate the nanoformulations. Each run of this protocol yielded 48 nanoformulations (16 unique formulations prepared in triplicate), occupying half of a 96-well plate.

### 5.2.3. Nanoformulation characterization

For high-throughput nanoformulation characterization, the OT-2’s 8-channel pipette was used to dilute each nanoformulation by mixing 20 µL of the prepared sample with 180 µL of deionized water (10× dilution) in a black-walled, transparent 96-well plate. Dynamic Light Scattering (DLS) measurements were performed using a DynaPro Plate Reader III (Wyatt Technology) to obtain particle diameter, polydispersity index (PDI), and sample quality scores.

### 5.2.4. Active learning study design

The active learning-based optimization campaign was conducted using the Ax library built on BoTorch [40]. Each formulation was featurized based on the input ratios of the drug and excipients to train Gaussian Process surrogate models. These models aimed to predict four key formulation performance targets: particle size, PDI, theoretical drug loading, and formulation complexity. Particle size and PDI were obtained from the DLS plate reader as described above, where particle sizes were reported in diameters and PDI reflected the formulation uniformity. Theoretical drug loading was calculated as the product of the initial drug input ratio and a binary sample quality score. This score was determined based on the analyzability of the formulation by the DLS plate reader: a value of 1 indicated the formulation was analyzable, while a value of 0 indicated it was not, usually caused by the presence of large precipitations within the samples. Formulation complexity was defined as the number of excipients used in a given formulation.

To facilitate the optimization campaign, performance metrics not naturally within the [0, 1] range were normalized. Specifically, particle size was linearly scaled between 0 (0 nm) and 1 (≥1000 nm), and formulation complexity was normalized by dividing by the maximum number of excipients (12). For formulations deemed non-analyzable (formulation quality = 0), particle size and PDI were set to 1, and theoretical drug loading was set to 0. The Bayesian optimization framework employed the qLogNoisyExpectedHypervolumeImprovement acquisition function to guide the selection of the most promising candidates for each subsequent iteration. The campaign began with a randomly selected initial iteration sampled from the full design space using Sobol method. After preparation and characterization, the data from each batch informed the selection of the next ten iterations with the goal to maximize theoretical drug loading while minimizing particle size, PDI, and formulation complexity. Each iteration includes 16 unique formulations, each prepared in triplicate. This batch size was chosen to accommodate the miniaturized experimental setup on a 96-well plate and to optimize the efficiency of DLS measurement time.

## **5.3. DOE-manual formulation screening**

### 5.3.1. DOE

Based on the results of the active learning-automation optimization campaign, the nanoformulation design space was refined to include four key factors (excipient types), each evaluated at three levels (excipient ratios). To systematically explore this refined design space, a DOE approach was implemented using the pyDOE2 package in Python. Instead of testing all 81 possible combinations from a full factorial design, the Generalized Subset Designs (GSD) method efficiently reduced the number of formulations to 20 representative candidates.

### 5.3.2. Benchtop nanoformulation preparation

A standard manual nanoprecipitation method was used for preparing and characterizing the 20 DOE-designed formulations. Aqueous and organic stock solutions were prepared in the same way as in the automation workflow. For benchtop-scale preparation, the sample volume was scaled up tenfold (from 300 µL to 3 mL) to yield a more representative sample size. Specifically, 2.7 mL of the aqueous phase was added to a 1-dram glass vial, followed by the addition of 0.3 mL of the corresponding organic phase using a pipette. The mixture was stirred using a magnetic stirrer at 2000 rpm for 10 minutes to induce nanoprecipitation. After mixing, the vials were covered with aluminum foil (with a few needle holes) and left in a fume hood overnight to allow solvent evaporation and nanoformulation hardening. Before characterization, the resulting nanosuspensions were filtered using a 0.45 µm syringe membrane filter to remove any drug or excipient precipitate.

### 5.3.3. Formulation characterization

Particle size and PDI of the benchtop-scale nanoformulations were measured using the same DLS plate reader as in the automation campaign. For drug concentration analysis, 100 µL of each nanoformulation was diluted in 900 µL of anhydrous ethanol, followed by ACF drug concentration measurement using a high performance liquid chromatography system (HPLC, Section 5.3.5). For stability testing, all 20 formulations were stored at room temperature in the dark in sealed 1-dram vials for two weeks. Formulations were classified as stable if no visible precipitation was observed. Stable formulations were then further analyzed for size, PDI, and drug concentration, using the same methods described above.

### 5.3.4. Solubility test

An excess amount of ACF was added to deionized water in 1-dram vials, sealed and stored in darkness overnight. After incubation, the saturated aqueous solutions were filtered using a 0.22 µm syringe membrane filter to remove undissolved drug. The filtered saturated solutions were then transferred to HPLC vials for concentration analysis. The solubility experiment was performed in triplicate.

### 5.3.5. HPLC

ACF quantification was performed using an Agilent 1260 Infinity II HPLC system (Agilent Technologies, US) equipped with a diode-array detector (DAD). Chromatographic separation was achieved on a ZORBAX Eclipse XDB-C18 column (4.6 × 150 mm, 5 µm) with a guard column. The mobile phase consisted of 20% (v/v) Milli-Q water and 80% (v/v) methanol, delivered at a flow rate of 1 mL/min under 40 degrees. Detection of ACF was conducted at a wavelength of 275 nm. Calibration curves were established over a ACF concentration range of 1–500 µg/mL, demonstrating linearity with R2 values above 0.99.

# **Data availability**

The dataset and results that support the findings of this study are available on GitHub (https://github.com/ZeqingBao/SDL5\_Nano).

# **Code availability**

The code that supports the findings of this study is available on GitHub (https://github.com/ZeqingBao/SDL5\_Nano).

# **References**

[1] Q. Liu, J. Zou, Z. Chen, W. He, W. Wu, Current research trends of nanomedicines, Acta Pharm. Sin. B 13 (2023) 4391–4416. https://doi.org/10.1016/j.apsb.2023.05.018.

[2] V. Weissig, T. Elbayoumi, B. Flühmann, A. Barton, The Growing Field of Nanomedicine and Its Relevance to Pharmacy Curricula, Am. J. Pharm. Educ. 85 (2021) 8331. https://doi.org/10.5688/ajpe8331.

[3] F. Farjadian, A. Ghasemi, O. Gohari, A. Roointan, M. Karimi, M.R. Hamblin, Nanopharmaceuticals and nanomedicines currently on the market: challenges and opportunities, Nanomed. 14 (2019) 93–126. https://doi.org/10.2217/nnm-2018-0120.

[4] W. Lu, J. Yao, X. Zhu, Y. Qi, Nanomedicines: Redefining traditional medicine, Biomed. Pharmacother. 134 (2021) 111103. https://doi.org/10.1016/j.biopha.2020.111103.

[5] D. Sivadasan, M.H. Sultan, O. Madkhali, Y. Almoshari, N. Thangavel, Polymeric Lipid Hybrid Nanoparticles (PLNs) as Emerging Drug Delivery Platform—A Comprehensive Review of Their Properties, Preparation Methods, and Therapeutic Applications, Pharmaceutics 13 (2021) 1291. https://doi.org/10.3390/pharmaceutics13081291.

[6] Y. Liu, Y. Liang, J. Yuhong, P. Xin, J.L. Han, Y. Du, X. Yu, R. Zhu, M. Zhang, W. Chen, Y. Ma, Advances in Nanotechnology for Enhancing the Solubility and Bioavailability of Poorly Soluble Drugs, Drug Des. Devel. Ther. 18 (2024) 1469–1495. https://doi.org/10.2147/DDDT.S447496.

[7] J.K. Patra, G. Das, L.F. Fraceto, E.V.R. Campos, M. del P. Rodriguez-Torres, L.S. Acosta-Torres, L.A. Diaz-Torres, R. Grillo, M.K. Swamy, S. Sharma, S. Habtemariam, H.-S. Shin, Nano based drug delivery systems: recent developments and future prospects, J. Nanobiotechnology 16 (2018) 71. https://doi.org/10.1186/s12951-018-0392-8.

[8] M.G. Papich, M.N. Martinez, Applying Biopharmaceutical Classification System (BCS) Criteria to Predict Oral Absorption of Drugs in Dogs: Challenges and Pitfalls, AAPS J. 17 (2015) 948–964. https://doi.org/10.1208/s12248-015-9743-7.

[9] Y. Yang, Z. Ye, Y. Su, Q. Zhao, X. Li, D. Ouyang, Deep learning for in vitro prediction of pharmaceutical formulations, Acta Pharm. Sin. B 9 (2019) 177–185. https://doi.org/10.1016/j.apsb.2018.09.010.

[10] J. Dong, Z. Wu, H. Xu, D. Ouyang, FormulationAI: a novel web-based platform for drug formulation design driven by artificial intelligence, Brief. Bioinform. 25 (2024) bbad419. https://doi.org/10.1093/bib/bbad419.

[11] Z. Bao, J. Kim, C. Kwok, F. Le Devedec, C. Allen, A dataset on formulation parameters and characteristics of drug-loaded PLGA microparticles, Sci. Data 12 (2025) 364. https://doi.org/10.1038/s41597-025-04621-9.

[12] A.J. Gormley, Machine learning in drug delivery, J. Controlled Release 373 (2024) 23–30. https://doi.org/10.1016/j.jconrel.2024.06.045.

[13] P. Bannigan, M. Aldeghi, Z. Bao, F. Häse, A. Aspuru-Guzik, C. Allen, Machine learning directed drug formulation development, Adv. Drug Deliv. Rev. 175 (2021) 113806. https://doi.org/10.1016/j.addr.2021.05.016.

[14] Z. Bao, J. Bufton, R.J. Hickman, A. Aspuru-Guzik, P. Bannigan, C. Allen, Revolutionizing drug formulation development: The increasing impact of machine learning, Adv. Drug Deliv. Rev. 202 (2023) 115108. https://doi.org/10.1016/j.addr.2023.115108.

[15] J.D. Murray, J.J. Lange, H. Bennett-Lenane, R. Holm, M. Kuentz, P.J. O’Dwyer, B.T. Griffin, Advancing algorithmic drug product development: Recommendations for machine learning approaches in drug formulation, Eur. J. Pharm. Sci. 191 (2023) 106562. https://doi.org/10.1016/j.ejps.2023.106562.

[16] Z. Bao, F. Yung, R.J. Hickman, A. Aspuru-Guzik, P. Bannigan, C. Allen, Data-driven development of an oral lipid-based nanoparticle formulation of a hydrophobic drug, Drug Deliv. Transl. Res. 14 (2024) 1872–1887. https://doi.org/10.1007/s13346-023-01491-9.

[17] H. Gao, W. Wang, J. Dong, Z. Ye, D. Ouyang, An integrated computational methodology with data-driven machine learning, molecular modeling and PBPK modeling to accelerate solid dispersion formulation design, Eur. J. Pharm. Biopharm. Off. J. Arbeitsgemeinschaft Pharm. Verfahrenstechnik EV 158 (2021) 336–346. https://doi.org/10.1016/j.ejpb.2020.12.001.

[18] J. Lange, A. Anelli, J. Alsenz, M. Kuentz, P. O’Dwyer, W. Saal, N. Wyttenbach, B. Griffin, Comparative Analysis of Chemical Descriptors by Machine Learning Reveals Atomistic Insights into Solute-Lipid Interactions, Mol. Pharm. 21 (2024) 3343–3355. https://doi.org/10.1021/acs.molpharmaceut.4c00080.

[19] Z. Bao, G. Tom, A. Cheng, J. Watchorn, A. Aspuru-Guzik, C. Allen, Towards the prediction of drug solubility in binary solvent mixtures at various temperatures using machine learning, J. Cheminformatics 16 (2024) 117. https://doi.org/10.1186/s13321-024-00911-3.

[20] J.J. Ong, B.M. Castro, S. Gaisford, P. Cabalar, A.W. Basit, G. Pérez, A. Goyanes, Accelerating 3D printing of pharmaceutical products using machine learning, Int. J. Pharm. X 4 (2022) 100120. https://doi.org/10.1016/j.ijpx.2022.100120.

[21] J. Deng, Z. Ye, W. Zheng, J. Chen, H. Gao, Z. Wu, G. Chan, Y. Wang, D. Cao, Y. Wang, S.M.-Y. Lee, D. Ouyang, Machine learning in accelerating microsphere formulation development, Drug Deliv. Transl. Res. 13 (2023) 966–982. https://doi.org/10.1007/s13346-022-01253-z.

[22] B. Castro, M. Elbadawi, J. Ong, T. Pollard, Z. Song, S. Gaisford, G. Pérez, A. Basit, P. Cabalar, A. Goyanes, Machine learning predicts 3D printing performance of over 900 drug delivery systems, J. Controlled Release 337 (2021) 530–545. https://doi.org/10.1016/j.jconrel.2021.07.046.

[23] A. Abostait, M. Abdelkarim, Z. Bao, Y. Miyake, W.H. Tse, C. Di Ciano-Oliveir, T. Buerki-Thurnherr, C. Allen, R. Keijzer, H.I. Labouta, Optimizing lipid nanoparticles for fetal gene delivery in vitro, ex vivo, and aided with machine learning, J. Controlled Release 376 (2024) 678–700.

[24] J.M. Schmitt, J.M. Baumann, M.M. Morgen, Predicting Spray Dried Dispersion Particle Size Via Machine Learning Regression Methods, Pharm. Res. 39 (2022) 3223–3239. https://doi.org/10.1007/s11095-022-03370-3.

[25] F. Wang, M. Elbadawi, S.L. Tsilova, S. Gaisford, A.W. Basit, M. Parhizkar, Machine learning predicts electrospray particle size, Mater. Des. 219 (2022) 110735. https://doi.org/10.1016/j.matdes.2022.110735.

[26] J. Wang, N. Heshmati Aghda, J. Jiang, A. Mridula Habib, D. Ouyang, M. Maniruzzaman, 3D bioprinted microparticles: Optimizing loading efficiency using advanced DoE technique and machine learning modeling, Int. J. Pharm. 628 (2022) 122302. https://doi.org/10.1016/j.ijpharm.2022.122302.

[27] S.H. Shetty, S. Shetty, C. Singh, A. Rao, Supervised Machine Learning: Algorithms and Applications, in: Fundam. Methods Mach. Deep Learn., John Wiley & Sons, Ltd, 2022: pp. 1–16. https://doi.org/10.1002/9781119821908.ch1.

[28] Z. Bao, Towards Data-driven Development of Advanced Drug Formulations Leveraging Machine Learning and Experimental Automation, Ph.D., University of Toronto (Canada), 2024. https://www.proquest.com/docview/3127430094/abstract/689863DE919B4F90PQ/1 (accessed March 27, 2025).

[29] T. Lookman, P.V. Balachandran, D. Xue, R. Yuan, Active learning in materials science with emphasis on adaptive sampling using uncertainties for targeted design, Npj Comput. Mater. 5 (2019) 1–17. https://doi.org/10.1038/s41524-019-0153-8.

[30] D. Reker, G. Schneider, Active-learning strategies in computer-assisted drug discovery, Drug Discov. Today 20 (2015) 458–465. https://doi.org/10.1016/j.drudis.2014.12.004.

[31] H. Narayanan, F. Dingfelder, A. Butté, N. Lorenzen, M. Sokolov, P. Arosio, Machine Learning for Biologics: Opportunities for Protein Engineering, Developability, and Formulation, Trends Pharmacol. Sci. 42 (2021) 151–165. https://doi.org/10.1016/j.tips.2020.12.004.

[32] R. J. Hickman, M. Sim, S. Pablo-García, G. Tom, I. Woolhouse, H. Hao, Z. Bao, P. Bannigan, C. Allen, M. Aldeghi, A. Aspuru-Guzik, Atlas: a brain for self-driving laboratories, Digit. Discov. (2025). https://doi.org/10.1039/D4DD00115J.

[33] R.J. Hickman, P. Bannigan, Z. Bao, A. Aspuru-Guzik, C. Allen, Self-driving laboratories: A paradigm shift in nanomedicine development, Matter 6 (2023) 1071–1081. https://doi.org/10.1016/j.matt.2023.02.007.

[34] M.J. Tamasi, R.A. Patel, C.H. Borca, S. Kosuri, H. Mugnier, R. Upadhya, N.S. Murthy, M.A. Webb, A.J. Gormley, Machine Learning on a Robotic Platform for the Design of Polymer–Protein Hybrids, Adv. Mater. 34 (2022) 2201809. https://doi.org/10.1002/adma.202201809.

[35] S. Lo, S. G. Baird, J. Schrier, B. Blaiszik, N. Carson, I. Foster, A. Aguilar-Granda, S. V. Kalinin, B. Maruyama, M. Politi, H. Tran, T. D. Sparks, A. Aspuru-Guzik, Review of low-cost self-driving laboratories in chemistry and materials science: the “frugal twin” concept, Digit. Discov. 3 (2024) 842–868. https://doi.org/10.1039/D3DD00223C.

[36] F. Häse, L.M. Roch, A. Aspuru-Guzik, Next-Generation Experimentation with Self-Driving Laboratories, Trends Chem. 1 (2019) 282–291. https://doi.org/10.1016/j.trechm.2019.02.007.

[37] T. Wu, S. Kheiri, R.J. Hickman, H. Tao, T.C. Wu, Z.-B. Yang, X. Ge, W. Zhang, M. Abolhasani, K. Liu, A. Aspuru-Guzik, E. Kumacheva, Self-driving lab for the photochemical synthesis of plasmonic nanoparticles with targeted structural and optical properties, Nat. Commun. 16 (2025) 1473. https://doi.org/10.1038/s41467-025-56788-9.

[38] F. Strieth-Kalthoff, H. Hao, V. Rathore, J. Derasp, T. Gaudin, N.H. Angello, M. Seifrid, E. Trushina, M. Guy, J. Liu, X. Tang, M. Mamada, W. Wang, T. Tsagaantsooj, C. Lavigne, R. Pollice, T.C. Wu, K. Hotta, L. Bodo, S. Li, M. Haddadnia, A. Wołos, R. Roszak, C.T. Ser, C. Bozal-Ginesta, R.J. Hickman, J. Vestfrid, A. Aguilar-Granda, E.L. Klimareva, R.C. Sigerson, W. Hou, D. Gahler, S. Lach, A. Warzybok, O. Borodin, S. Rohrbach, B. Sanchez-Lengeling, C. Adachi, B.A. Grzybowski, L. Cronin, J.E. Hein, M.D. Burke, A. Aspuru-Guzik, Delocalized, asynchronous, closed-loop discovery of organic laser emitters, Science 384 (2024) eadk9227. https://doi.org/10.1126/science.adk9227.

[39] G. Tom, S.P. Schmid, S.G. Baird, Y. Cao, K. Darvish, H. Hao, S. Lo, S. Pablo-García, E.M. Rajaonson, M. Skreta, N. Yoshikawa, S. Corapi, G.D. Akkoc, F. Strieth-Kalthoff, M. Seifrid, A. Aspuru-Guzik, Self-Driving Laboratories for Chemistry and Materials Science, Chem. Rev. 124 (2024) 9633–9732. https://doi.org/10.1021/acs.chemrev.4c00055.

[40] BoTorch | BoTorch, (n.d.). https://botorch.org/ (accessed April 22, 2025).

[41] M.M. El-Hammadi, J.L. Arias, Recent Advances in the Surface Functionalization of PLGA-Based Nanomedicines, Nanomaterials 12 (2022) 354. https://doi.org/10.3390/nano12030354.

[42] D. Zhang, L. Liu, J. Wang, H. Zhang, Z. Zhang, G. Xing, X. Wang, M. Liu, Drug-loaded PEG-PLGA nanoparticles for cancer treatment, Front. Pharmacol. 13 (2022). https://doi.org/10.3389/fphar.2022.990505.

[43] C. Viegas, A.B. Patrício, J.M. Prata, A. Nadhman, P.K. Chintamaneni, P. Fonte, Solid Lipid Nanoparticles vs. Nanostructured Lipid Carriers: A Comparative Review, Pharmaceutics 15 (2023) 1593. https://doi.org/10.3390/pharmaceutics15061593.

[44] J. Akbari, Saeedi ,Majid, Ahmadi ,Fatemeh, Hashemi ,Seyyed Mohammad Hassan, Babaei ,Amirhossein, Yaddollahi ,Sadra, Rostamkalaei ,Seyyed Sohrab, Asare-Addo ,Kofi, A. and Nokhodchi, Solid lipid nanoparticles and nanostructured lipid carriers: a review of the methods of manufacture and routes of administration, Pharm. Dev. Technol. 27 (2022) 525–544. https://doi.org/10.1080/10837450.2022.2084554.

[45] R. Al-Kassas, M. Bansal, J. Shaw, Nanosizing techniques for improving bioavailability of drugs, J. Controlled Release 260 (2017) 202–212. https://doi.org/10.1016/j.jconrel.2017.06.003.

[46] Z.H. Mok, The effect of particle size on drug bioavailability in various parts of the body, Pharm. Sci. Adv. 2 (2024) 100031. https://doi.org/10.1016/j.pscia.2023.100031.

[47] R. Kumar, A.K. Thakur, P. Chaudhari, N. Banerjee, Particle Size Reduction Techniques of Pharmaceutical Compounds for the Enhancement of Their Dissolution Rate and Bioavailability, J. Pharm. Innov. 17 (2022) 333–352. https://doi.org/10.1007/s12247-020-09530-5.

[48] J. Antony, Design of Experiments for Engineers and Scientists, Elsevier, 2023.

[49] A.C. Correia, J.N. Moreira, J.M. Sousa Lobo, A.C. Silva, Design of experiment (DoE) as a quality by design (QbD) tool to optimise formulations of lipid nanoparticles for nose-to-brain drug delivery, Expert Opin. Drug Deliv. 20 (2023) 1731–1748. https://doi.org/10.1080/17425247.2023.2274902.

[50] B. Singh, Kapil ,Rishi, Nandi ,Mousumi, N. and Ahuja, Developing oral drug delivery systems using formulation by design: vital precepts, retrospect and prospects, Expert Opin. Drug Deliv. 8 (2011) 1341–1360. https://doi.org/10.1517/17425247.2011.605120.

[51] S.P.S. Teja, N. Damodharan, 23 Full Factorial Model for Particle Size Optimization of Methotrexate Loaded Chitosan Nanocarriers: A Design of Experiments (DoE) Approach, BioMed Res. Int. 2018 (2018) 7834159. https://doi.org/10.1155/2018/7834159.

[52] R. Gupta, H. Xie, M. Sarkar, Y. Chen, Design of Experiment (DOE) for Optimization of PLGA Nanoparticles, FASEB J. 36 (2022). https://doi.org/10.1096/fasebj.2022.36.S1.R6197.

[53] L. Gurba-Bryśkiewicz, W. Maruszak, D.A. Smuga, K. Dubiel, M. Wieczorek, Quality by Design (QbD) and Design of Experiments (DOE) as a Strategy for Tuning Lipid Nanoparticle Formulations for RNA Delivery, Biomedicines 11 (2023) 2752. https://doi.org/10.3390/biomedicines11102752.

[54] Y. Qin, A.A. Walters, N. Rouatbi, J.T.-W. Wang, H.M. Abdel-Bar, K.T. Al-Jamal, Evaluation of a DoE based approach for comprehensive modelling of the effect of lipid nanoparticle composition on nucleic acid delivery, Biomaterials 299 (2023) 122158. https://doi.org/10.1016/j.biomaterials.2023.122158.

[55] J. Salazar, O. Heinzerling, R.H. Müller, J.P. Möschwitzer, Process optimization of a novel production method for nanosuspensions using design of experiments (DoE), Int. J. Pharm. 420 (2011) 395–403. https://doi.org/10.1016/j.ijpharm.2011.09.003.

[56] T. Liu, X. Yu, H. Yin, Study of Top-down and Bottom-up Approaches by Using Design of Experiment (DoE) to Produce Meloxicam Nanocrystal Capsules, AAPS PharmSciTech 21 (2020) 79. https://doi.org/10.1208/s12249-020-1621-7.

[57] X. Ding, R. Cui, J. Yu, T. Liu, T. Zhu, D. Wang, J. Chang, Z. Fan, X. Liu, K. Chen, H. Jiang, X. Li, X. Luo, M. Zheng, Active Learning for Drug Design: A Case Study on the Plasma Exposure of Orally Administered Drugs, J. Med. Chem. 64 (2021) 16838–16853. https://doi.org/10.1021/acs.jmedchem.1c01683.

[58] R.A. Patel, S.S. Kesharwani, F. Ibrahim, Active learning and Gaussian processes for the development of dissolution models: An AI-based data-efficient approach, J. Controlled Release 379 (2025) 316–326. https://doi.org/10.1016/j.jconrel.2025.01.003.

[59] L.K. Vora, A.D. Gholap, K. Jetha, R.R.S. Thakur, H.K. Solanki, V.P. Chavda, Artificial Intelligence in Pharmaceutical Technology and Drug Delivery Design, Pharmaceutics 15 (2023) 1916. https://doi.org/10.3390/pharmaceutics15071916.

[60] J. Vamathevan, D. Clark, P. Czodrowski, I. Dunham, E. Ferran, G. Lee, B. Li, A. Madabhushi, P. Shah, M. Spitzer, S. Zhao, Applications of machine learning in drug discovery and development, Nat. Rev. Drug Discov. 18 (2019) 463–477. https://doi.org/10.1038/s41573-019-0024-5.

[61] E. Egorov, C. Pieters, H. Korach-Rechtman, J. Shklover, A. Schroeder, Robotics, microfluidics, nanotechnology and AI in the synthesis and evaluation of liposomes and polymeric drug delivery systems, Drug Deliv. Transl. Res. 11 (2021) 345–352. https://doi.org/10.1007/s13346-021-00929-2.

[62] N. Serov, V. Vinogradov, Artificial intelligence to bring nanomedicine to life, Adv. Drug Deliv. Rev. 184 (2022) 114194. https://doi.org/10.1016/j.addr.2022.114194.

[63] M. Hammel, Y. Fan, A. Sarode, A.E. Byrnes, N. Zang, P. Kou, K. Nagapudi, D. Leung, C.C. Hoogenraad, T. Chen, C.-W. Yen, G.L. Hura, Correlating the Structure and Gene Silencing Activity of Oligonucleotide-Loaded Lipid Nanoparticles Using Small-Angle X-ray Scattering, ACS Nano 17 (2023) 11454–11465. https://doi.org/10.1021/acsnano.3c01186.

[64] A. Pratsinis, Y. Fan, M. Portmann, M. Hammel, P. Kou, A. Sarode, P. Ringler, L. Kovacik, M.E. Lauer, J. Lamerz, G.L. Hura, C.-W. Yen, M. Keller, Impact of non-ionizable lipids and phase mixing methods on structural properties of lipid nanoparticle formulations, Int. J. Pharm. 637 (2023) 122874. https://doi.org/10.1016/j.ijpharm.2023.122874.

[65] Y. Fan, Z. Shi, S. Ma, S.Z.A. Razvi, Y. Fu, T. Chen, J. Gruenhagen, K. Zhang, Spectroscopy-Based Local Modeling Method for High-Throughput Quantification of Nucleic Acid Loading in Lipid Nanoparticles, Anal. Chem. 94 (2022) 9081–9090. https://doi.org/10.1021/acs.analchem.2c01346.

[66] A. Sarode, Y. Fan, A.E. Byrnes, M. Hammel, G.L. Hura, Y. Fu, P. Kou, C. Hu, F.I. Hinz, J. Roberts, S.G. Koenig, K. Nagapudi, C.C. Hoogenraad, T. Chen, D. Leung, C.-W. Yen, Predictive high-throughput screening of PEGylated lipids in oligonucleotide-loaded lipid nanoparticles for neuronal gene silencing, Nanoscale Adv. 4 (2022) 2107–2123. https://doi.org/10.1039/D1NA00712B.

[67] L. Cui, M.R. Hunter, S. Sonzini, S. Pereira, S.M. Romanelli, K. Liu, W. Li, L. Liang, B. Yang, N. Mahmoudi, A.S. Desai, Mechanistic Studies of an Automated Lipid Nanoparticle Reveal Critical Pharmaceutical Properties Associated with Enhanced mRNA Functional Delivery In Vitro and In Vivo, Small 18 (2022) 2105832. https://doi.org/10.1002/smll.202105832.

[68] L. Cui, S. Pereira, S. Sonzini, S. van Pelt, S.M. Romanelli, L. Liang, D. Ulkoski, V.R. Krishnamurthy, E. Brannigan, C. Brankin, A.S. Desai, Development of a high-throughput platform for screening lipid nanoparticles for mRNA delivery, Nanoscale 14 (2022) 1480–1491. https://doi.org/10.1039/D1NR06858J.

[69] Y. Fan, C.-W. Yen, H.-C. Lin, W. Hou, A. Estevez, A. Sarode, A. Goyon, J. Bian, J. Lin, S.G. Koenig, D. Leung, K. Nagapudi, K. Zhang, Automated high-throughput preparation and characterization of oligonucleotide-loaded lipid nanoparticles, Int. J. Pharm. 599 (2021) 120392. https://doi.org/10.1016/j.ijpharm.2021.120392.

[70] D.M. Loy, R. Krzysztoń, U. Lächelt, J.O. Rädler, E. Wagner, Controlling Nanoparticle Formulation: A Low-Budget Prototype for the Automation of a Microfluidic Platform, Processes 9 (2021) 129. https://doi.org/10.3390/pr9010129.

[71] J.T. Goodman, A.S. Mullis, L. Dunshee, A. Mitra, B. Narasimhan, Automated High-Throughput Synthesis of Protein-Loaded Polyanhydride Nanoparticle Libraries, ACS Comb. Sci. 20 (2018) 298–307. https://doi.org/10.1021/acscombsci.8b00008.

[72] K. Miladi, S. Sfar, H. Fessi, A. Elaissari, Nanoprecipitation Process: From Particle Preparation to In Vivo Applications, in: C. Vauthier, G. Ponchel (Eds.), Polym. Nanoparticles Nanomedicines Guide Their Des. Prep. Dev., Springer International Publishing, Cham, 2016: pp. 17–53. https://doi.org/10.1007/978-3-319-41421-8\_2.

[73] J. Aubry, F. Ganachaud, J.-P. Cohen Addad, B. Cabane, Nanoprecipitation of Polymethylmethacrylate by Solvent Shifting:1. Boundaries, Langmuir 25 (2009) 1970–1979. https://doi.org/10.1021/la803000e.

[74] E. Lepeltier, C. Bourgaux, P. Couvreur, Nanoprecipitation and the “Ouzo effect”: Application to drug delivery devices, Adv. Drug Deliv. Rev. 71 (2014) 86–97. https://doi.org/10.1016/j.addr.2013.12.009.

[75] V.-A. Duong, T.-T.-L. Nguyen, H.-J. Maeng, Preparation of Solid Lipid Nanoparticles and Nanostructured Lipid Carriers for Drug Delivery and the Effects of Preparation Parameters of Solvent Injection Method, Mol. Basel Switz. 25 (2020) E4781. https://doi.org/10.3390/molecules25204781.

[76] S. Schubert, J. Joseph T.  Delaney, U. S. Schubert, Nanoprecipitation and nanoformulation of polymers : from history to powerful possibilities beyond poly(lactic acid), Soft Matter 7 (2011) 1581–1588. https://doi.org/10.1039/C0SM00862A.

[77] S. Hornig, T. Heinze, C. Remzi Becer, U. S. Schubert, Synthetic polymeric nanoparticles by nanoprecipitation, J. Mater. Chem. 19 (2009) 3838–3840. https://doi.org/10.1039/B906556N.

[78] H.M. Xia, Y.P. Seah, Y.C. Liu, W. Wang, A.G.G. Toh, Z.P. Wang, Anti-solvent precipitation of solid lipid nanoparticles using a microfluidic oscillator mixer, Microfluid. Nanofluidics 19 (2015) 283–290. https://doi.org/10.1007/s10404-014-1517-5.

[79] A. Iqbal, M. Zaman, M. Wahab Amjad, S. Adnan, M. Abdul Ghafoor Raja, S.F. Haider Rizvi, M.W. Mustafa, U. Farooq, G. Abbas, S. Shah, Solid Lipid Nanoparticles of Mycophenolate Mofetil: An Attempt to Control the Release of an Immunosuppressant, Int. J. Nanomedicine 15 (2020) 5603–5612. https://doi.org/10.2147/IJN.S255636.

[80] T. Jiang, Han ,Ning, Zhao ,Buwen, Xie ,Yuling, S. and Wang, Enhanced dissolution rate and oral bioavailability of simvastatin nanocrystal prepared by sonoprecipitation, Drug Dev. Ind. Pharm. 38 (2012) 1230–1239. https://doi.org/10.3109/03639045.2011.645830.

[81] T. Chen, Y. Peng, M. Qiu, C. Yi, Z. Xu, Recent advances in mixing-induced nanoprecipitation: from creating complex nanostructures to emerging applications beyond biomedicine, Nanoscale 15 (2023) 3594–3609. https://doi.org/10.1039/D3NR00280B.

[82] V. Dave, K. Tak, A. Sohgaura, A. Gupta, V. Sadhu, K.R. Reddy, Lipid-polymer hybrid nanoparticles: Synthesis strategies and biomedical applications, J. Microbiol. Methods 160 (2019) 130–142. https://doi.org/10.1016/j.mimet.2019.03.017.

[83] M. Zhang, J. He, W. Zhang, J. Liu, Fabrication of TPGS-Stabilized Liposome-PLGA Hybrid Nanoparticle Via a New Modified Nanoprecipitation Approach: In Vitro and In Vivo Evaluation, Pharm. Res. 35 (2018) 199. https://doi.org/10.1007/s11095-018-2485-3.

[84] K. Takayama, M. Fujikawa, Y. Obata, M. Morishita, Neural network based optimization of drug formulations, Adv. Drug Deliv. Rev. 55 (2003) 1217–1231. https://doi.org/10.1016/s0169-409x(03)00120-0.

[85] Z. Li, B.R. Cho, B.J. Melloy, Quality by Design Studies on Multi-response Pharmaceutical Formulation Modeling and Optimization, J. Pharm. Innov. 8 (2013) 28–44. https://doi.org/10.1007/s12247-012-9145-7.

[86] S. Sathasivam, M. Mahabooba, M. Jeevanantham, D. Satishkumar, An Review on Optimization Technique in Pharmaceutical Formulation and ProcessingHandbook of research in Big Data Analytics, Artificial intelligence and Machine learning CHAPTER 2 An review on Optimization Technique in Pharmaceutical Formulation and Processing Quantitative Structure-Activity Relationship (QSAR) Modeling, in: 2024.

[87] A. Bose, D. Roy Burman, B. Sikdar, P. Patra, Nanomicelles: Types, properties and applications in drug delivery, IET Nanobiotechnol. 15 (2021) 19–27. https://doi.org/10.1049/nbt2.12018.

[88] S. Sinha, A.K. Tripathi, A. Pandey, P. Naik, A. Pandey, V.S. Verma, Self-assembled PEGylated micelles for precise and targeted drug delivery: Current challenges and future directions, Biocatal. Agric. Biotechnol. 60 (2024) 103296. https://doi.org/10.1016/j.bcab.2024.103296.

[89] G.-W. Jin, N.S. Rejinold, J.-H. Choy, Multifunctional Polymeric Micelles for Cancer Therapy, Polymers 14 (2022) 4839. https://doi.org/10.3390/polym14224839.

[90] Z. Ahmad, A. Shah, M. Siddiq, H.-B. Kraatz, Polymeric micelles as drug delivery vehicles, RSC Adv. 4 (2014) 17028–17038. https://doi.org/10.1039/C3RA47370H.

[91] D. Nakmode, V. Bhavana, P. Thakor, J. Madan, P.K. Singh, S.B. Singh, J.M. Rosenholm, K.K. Bansal, N.K. Mehra, Fundamental Aspects of Lipid-Based Excipients in Lipid-Based Product Development, Pharmaceutics 14 (2022) 831. https://doi.org/10.3390/pharmaceutics14040831.

[92] A.M. Kassem, Ibrahim ,Hany M., A.M. and Samy, Development and optimisation of atorvastatin calcium loaded self-nanoemulsifying drug delivery system (SNEDDS) for enhancing oral bioavailability: in vitro and in vivo evaluation, J. Microencapsul. 34 (2017) 319–333. https://doi.org/10.1080/02652048.2017.1328464.

[93] F.-P. Schmied, A. Bernhardt, A. Engel, S. Klein, A Customized Screening Tool Approach for the Development of a Self-Nanoemulsifying Drug Delivery System (SNEDDS), AAPS PharmSciTech 23 (2021) 39. https://doi.org/10.1208/s12249-021-02176-7.

# **Acknowledgements**

This research was undertaken thanks in part to funding provided to the University of Toronto’s Acceleration Consortium from the Canada First Research Excellence Fund (CFREF-2022-00042). The authors gratefully acknowledge Dr. Sterling G. Baird for his valuable discussions and support in active learning and data visualization.

# **Author information**

**Acceleration Consortium, University of Toronto, Ontario, Canada**

Zeqing Bao & Frantz Le Devedec

**Leslie Dan Faculty of Pharmacy, University of Toronto, Toronto, Ontario, Canada**

Steven Huynh

Contributions

Z.B. led the conceptualization of this work, led the computational studies, performed the experimental studies, led the data analysis, and wrote the first draft of the paper. F.L.D. assisted with the data analysis and assisted with the review/editing of the paper. S.H. assisted with the data visualization and assisted with the review/editing of the paper.

Corresponding authors

Correspondence to Zeqing Bao (zeqing.bao@utoronto.ca).

# **Ethics declarations**

Competing interests

The authors declare no competing interests.