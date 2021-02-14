- Visual Planning Task, V2 dataset
  - Environment and problem set up
  - Dataset details
  - Experiments: Need to clearly find and document baseline, our method variants and metrics
    - May start with https://docs.google.com/presentation/d/1gh70xzSKQs543I56Z7YEDQPFn9tVXYA2UYKKmU3NKNY/edit#slide=id.g81fd2ea35e_0_0 and work backwards...

- Mujoco experiments, Phase 1, large dataset
  - Problem set up and environment / action space description
  - Need to find the experiments and results

- Mujoco experiments, Phase 2, fixed goal dataset
  - Additional restrictions on problem setup
  - Small dataset for policy, large dataset for auto encoder
  - Experiments (the latest ones, easy to find)


-------------------------------------------------------------
Visual Planning Experiments mapping from the above presentation:

Expt1 : 06-20-06-28-50-FixImagEncAndPolicyTrainD4EnvDec-V2-LatentPolicyNet-CE
Expt2 : 06-20-09-51-51-FineTuneEndtoEndAfterDecoderTraining-V2-LatentPolicyNet-CE
Expt3 : 06-20-08-11-22-FixImagEncAndPolicyTrainD4EnvDec-V2-LatentPolicyNet-CE
Expt4 : 06-20-08-48-19-FineTuneEndtoEndAfterDecoderTraining-V2-LatentPolicyNet-CE

Expt5 : 06-22-11-53-12-Fix_ImagEnc_AugmentedPolicy_Train_EnvDec-V2-LatentPolicyNet-CE
Expt6 : 06-22-12-13-04-Finetune-0622115312-V2-LatentPolicyNet-CE
Expt7 : 06-22-12-56-31-Finetune-0616203357-V2-LatentPolicyNet-CE
Expt8 : 06-22-13-14-41-Finetune-0616232643-V2-LatentPolicyNet-CE

-------------------------------------------------------------
Research papers read:
- Thourough
  - Robot Motion Planning in Learned Latent Spaces
    -  Uses latent learning to improve steps in motion planning (not so much related work at all actually)
  - End to End Learning for Self-Driving Cars
    - Behavior cloning (the base line), does not use latent representation or auto encoders 
  - Embed to Control: A Locally Linear Latent Dynamics Model for Control from Raw Images
    - Our work can be seen as taking the next step of this. Learns the latent representation guided by dynamics of system but not the policy.
  - End-to-end training of deep visuomotor policies
    - End to end learning method, does not use latent representions.
  - Self supervised correspondence in VisuoMotor Policy Learning
    - Uses a diffrent cue for learning latent representations (visual correspondence)
- Skim
  - Fitting a Linear Control Policy to Demonstrations with a Kalman Constraint
  - Improving Sample Efficiency in Model-Free Reinforcement Learning from Images
  - Sim-to-Real via Sim-to-Sim: Data-efficient Robotic Grasping via Randomized-to-Canonical Adaptation Networks
  - Sim-to-Real Transfer of Robotic Control with Dynamics Randomization
-------------------------------------------------------------


\section{Introduction}
% why this problem is important (or) what are open issues in current state of the art
% general high level description of what we propose to do
% quick description of results

\section{Related Work}
% end-to-end learning
% auto encoders
% other items

\section{Methodology}
% write in DETAIL about our method
% encoders and policies
% comparison of trials for different choices we made (e.g. number of channels, compression ratios for autoencoders)

\section{Experiments}
% visual planning (2D planning)
% 3dof robot

\section{Conclusion}
% with these results, we can do image-to-policy networks. The results prove this.
% Future work includes <blah> and <blah>
% ciao!