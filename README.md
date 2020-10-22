# CrossIDsystem
An effective method of transfer learning is to constrain the trainable parameters to affine coefficients in the normalized layer to achieve the purpose of transfer learning. When the domain difference is not big, the effect of this method is obvious. Although the data may not be the same as those visited, I believe it can be applied to most of the work. After all, who doesn't want a better BN layer?
\subsection{ID based transfer system}

ID in index form is the over compressed state of the identity attribute. Due to over compression, it loses its value and continuity, which makes it almost impossible for neural networks to understand and infer. Interestingly, the word embedding model, which is widely used in natural language processing, provides an idea to reconstruct index into higher-order features through supervised learning. Similar algorithms have also proposed in the image style transfer domain. Among them, CIN which learn affine parameters matching to each style enable a static neural network to generate images with multiple styles. These affine coefficients, also named high-level feature,  are the the modeling of image styles seen in the training process. Referring to the above algorithm, we belive that the identity attributes can be modeled as the inter-relationship of affine parameters. Therefore, the problem of cross identity recognition becomes find affine parameters matching to target. With this in mind, the ID based transfer system proposed. In the pre training stage, pretext classification task is used to supervise the backbone classifier in the system. At the same time, various high-order features bound to target are implant into backbone network so as to build the compatibility of ID changes. According to target index $c\in{\{1,...,C\}}$,$C\in\mathbb{R}$, normalization layers in our system fetch corresponding $\gamma_c$, $\beta_c$ and fusion them with the input feature map. Its forward propagation process can be denote as:
\begin{equation}
z=\gamma_c(\frac{x-\mu(x)}{\sigma(x)})+\beta_c
\end{equation}\label{eq4}
where the $\mu$ and $\sigma$ denote the mean and standard taken across both spatial and feature dimension. In transfer stage, the backbone classifier is frozen, while the random initialized ID is adopted and optimized with the supervision of classification loss. Finally, all the affine parameters matching to each target are saved to the database of the system. The  entire process of CrossID system can be regarded as a  generalized Expectation-Maximization (EM) system, where the pre train process is the  “Expectation step” and the following ID gathering step is its “Maximization step”. The method of freezing one part and estimating the other part enables the system to break through the original data boundary without producing redundant parameters. These two steps enables the system to break through the original data boundary without parameter duplicate.
\subsection{Some Common Mistakes}
