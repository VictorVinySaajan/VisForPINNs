import define1 from "./7006773330720914@1064.js";
import define2 from "./115f2c4ec1a42d7e@373.js";
import define3 from "./e2d1fe5a0e76c0da@29.js";

function _1(md){return(
md`<div style="text-align: center; font-size: 20px;"> 
    <h1> VisForPINNs </h1>
</div>

<div style="text-align: center;"> 
    <h1> Visualization for Understanding Physics Informed Neural Networks</h1>
</div>

<div style="text-align: center; font-size: 8px;">
    <h1> By: 
        <a href="https://www.itwm.fraunhofer.de/en/departments/tv/staff/viny_saajan_victor.html" style="color: blue;">Viny Saajan Victor</a>,
        <a href="https://www.itwm.fraunhofer.de/en/departments/tv/staff/manuel-ettmueller.html" style="color: blue;">Manuel Ettmüller</a>,
        <a href="https://www.itwm.fraunhofer.de/en/departments/tv/staff/andre-schmeisser.html" style="color: blue;">Dr. Andre Schmeißer</a>,
        <a href="https://vis.uni-kl.de/team/leitte/" style="color: blue;">Prof. Dr. Heike Leitte</a>, and
        <a href="https://www.itwm.fraunhofer.de/en/departments/tv/staff/simone-gramsch.html" style="color: blue;">Prof. Dr. Simone Gramsch</a>
    </h1>
</div>

<div style="text-align: center; font-size: 8px;">
    <h1> July 21, 2023 </h1>
</div>`
)}

function _2(md){return(
md`<div style="text-align: left; font-size: 14px;">
**NOTE:**  This "explainable" works best with the firefox browser. If using other browsers, please refresh <br>
the page if the plots don’t load.
</div>`
)}

function _3(md){return(
md`## Abstract

In this "explainable", we present the concept of Informed Machine Learning, which incorporates prior knowledge into the machine learning system. As part of this framework, we introduce Physics Informed Neural Networks (PINNs) as a complementary approach that is specifically designed to incorporate physics principles, in contrast to traditional data-driven neural networks. By utilizing interactive visualizations, we provide an intuitive understanding of the behavior of physics-driven networks compared to data-driven networks. Through an examination of various aspects, with a specific focus on the application of the industrial melt-spinning process, we explore the synergies between these networks by comparing the different loss terms that are used to train these networks. Subsequently, we present a hybrid network that leverages a combination of data-driven and physics-driven approaches. We contend that the incorporation of fundamental physics principles into data-driven machine learning systems serves to augment their dependability and overall performance.`
)}

function _4(cite,md){return(
md`## Introduction

Differential equations stand as a powerful mathematical tool widely utilized for comprehending and predicting the behavior of dynamic systems in nature, engineering, and society. The study of differential equations entails learning how to solve them and interpret the solutions.
Despite the fact that analytical methods for solving differential equations yield exact answers, their application becomes intricate when applied to complex problems. Consequently, several numerical methods have become popular, leveraging the development of computing capabilities. These methods provide approximate solutions that have sufficient accuracy for engineering purposes. Thus, they have evolved into the fundamental building blocks of many simulations, serving as surrogate models for their corresponding physical models. However, numerical simulations prove impractical for real-time applications involving scenarios that demand multiple queries due to their computational performance. As a remedy, supervised machine learning models are employed to learn and predict the behaviors exhibited in the simulations, making them suitable for real-time usage ${cite("c14")}.

In recent years, the realm of supervised machine learning (ML) has witnessed significant progress in various fields, such as computer vision and natural language processing. This progress can be attributed to the capacity of supervised ML to unveil underlying patterns and structures within the provided data. However, challenges persist, especially in terms of generalizability, reliability, stability, and expressiveness, particularly when dealing with limited training data. Let us delve into the functioning of traditional supervised machine learning models. These models require labeled data for training, where input data is paired with corresponding output labels. The primary goal here is to establish a connection between the input data and the desired output labels. These models are often referred to as "data-driven" since their effectiveness relies on the quality and quantity of the labeled training data. By learning from these labeled examples, the models gain the ability to identify patterns and make predictions on new, unseen data, thus demonstrating their generalization capabilities.`
)}

async function _5(html,FileAttachment){return(
html`<img src="${await FileAttachment("DD_ML_rev.png").url()}" style="width: 600px"/>`
)}

function _6(cite,md){return(
md`<em> The figure shows the information flow in data-driven machine learning. Figure adapted from ${cite("c1")} </em> `
)}

function _7(cite,md){return(
md`Insufficient data poses limitations for data-driven ML models. When the available training data fails to adequately represent the variability and capture the system behavior being examined, the resulting ML model will exhibit poor performance. Moreover, if the data contains noise and there are no means to impose constraints on the model other than through the data itself, the model's reliability will be compromised. Additionally, the model's explainability will be reduced, as it primarily focuses on mapping input-output data without providing insightful explanations. To address these limitations, one approach is to integrate prior knowledge into the machine learning process. This research field is commonly referred to as "Informed Machine Learning ${cite("c1")}." By incorporating prior knowledge, the ML models can overcome the challenges posed by insufficient data, noisy inputs, and limited explainability and model reliability.`
)}

async function _8(html,FileAttachment){return(
html`<img src="${await FileAttachment("IML_rev.png").url()}" style="width: 600px"/>`
)}

function _9(cite,md){return(
md`<em> The figure shows the information flow in data-driven machine learning and informed machine learning. Figure adapted from ${cite("c1")} </em>`
)}

function _10(md){return(
md`There are various ways to represent prior knowledge, and it can be integrated into different stages of the machine learning pipeline depending on data availability, knowledge sources, and application scenarios.`
)}

async function _11(html,FileAttachment){return(
html`<img src="${await FileAttachment("Sankey.png").url()}" style="width: 600px"/>`
)}

function _12(cite,md){return(
md`<em> The figure displays the sources, representations, and of prior knowledge and their integration into the ML pipeline. Figure adapted from ${cite("c1")} </em>`
)}

function _13(md){return(
md`## Physics-Informed Neural Networks (PINNs)`
)}

function _14(md){return(
md`Physics-informed neural networks (PINNs) are neural networks that incorporate prior scientific knowledge, such as differential equations, into the learning algorithm of the machine learning pipeline. These networks jointly learn to fit the training data while reducing the residual of the governing differential equations that describe the underlying physics of the model being examined. This process effectively constrains the model to adhere to the known laws of physics.`
)}

async function _15(html,FileAttachment){return(
html`<img src="${await FileAttachment("Sankey_PINN.png").url()}" style="width: 600px"/>`
)}

function _16(cite,md){return(
md`<em> The figure displays the source, representation, and integration of knowledge in PINNs. Figure adapted from ${cite("c1")} </em>`
)}

function _17(cite,md){return(
md`Raissi et al. spearheaded the introduction of Physics-Informed Neural Networks (PINNs) as an innovative category of solvers ${cite("c2", "c3")}. Since then, the domain of Physics-Informed Neural Networks (PINNs) is rapidly expanding, encompassing a wide range of equations such as ordinary differential equations (ODEs) ${cite("c4")}, partial differential equations (PDEs) ${cite("c2")}, fractional equations (FEs) ${cite("c5")}, integro-differential equations (IDEs) ${cite("c6")}, and stochastic differential equations (SDEs) ${cite("c7")}. This article focuses specifically on the utilization of PINNs for solving ODEs. The primary objective of this article is to delve into the inner workings of PINNs and explore the potential synergies between data-driven and physics-driven models. We aim to build robust, reliable, and informed machine learning models by employing an industrial application example of the melt spinning process.`
)}

function _18(md){return(
md`## Application: Ordinary Differential Equations for the Simulation of Melt Spinning Processes`
)}

function _19(md){return(
md`Melt spinning is a manufacturing process employed for the production of industrial fibers. In this process, a molten polymer is transformed into continuous filament fibers by extruding it through small openings known as spinnerets. These fibers possess desirable attributes such as strength, durability, wrinkle resistance, and moisture-wicking capabilities. Hence they find their applications across diverse fields, encompassing liquid and gas filtration, including applications like vacuum cleaner bags and water filtration systems. Additionally, they are utilized in insulation for roofing, flooring, and wall materials, automotive purposes involving seat covers, door panels, and headliners, as well as medical applications such as surgical gowns, masks, and drapes. Moreover, these fibers are integral to hygiene products such as diapers, sanitary pads, and wipes, and play essential roles in technologies such as batteries, fuel cells, and various other domains.`
)}

async function _20(html,FileAttachment){return(
html`<img src="${await FileAttachment("process_melt_spinning_simple-1.png").url()}" style= "width: 600px"/>`
)}

function _21(md){return(
md`<em> Figure depicting the industrial melt spinning process. </em>`
)}

function _22(cite,md){return(
md`Optimizing the melt spinning production process while maintaining the desired quality of the fibers involves analyzing the different properties of the fibers along their length. However, achieving real-time analysis is challenging due to the stochastic nature of the spinning processes. Consequently, modeling spinning processes ${cite("c8")} often entails a combination of differential equations that describe the fiber properties. Furthermore, specific fiber properties remain fixed at the start and end positions of the fibers due to the process conditions, resulting in differential equations with boundary conditions.`
)}

function _23(cite,tex,md){return(
md`Our specific use case in this analysis is the iso-thermal uniaxial spinning ${cite("c9")}. In this case, we examine a straight-line fiber that extends between points <em>r<sub>a</sub></em> and <em>r<sub>b</sub></em>, with a total fiber length of <em>L</em>. The unknown variables of the process here are the fiber velocity ${tex`u`} and the fiber tension ${tex`N`} along the spinline. While the velocity at the inlet and outlet are known due to the process setup, the profiles along the spinline are of interest for the engineers. Too much fiber tension or very steep velocity and tension gradients can cause damage to the final product and negatively influence the fiber properties. The system of ordinary differential equations (ODEs) governing the velocity <em>u</em> and tension <em>N</em> along the spinline is expressed as follows:`
)}

function _24(tex){return(
tex.block`\tag{1} 
\frac{du}{dx} = \frac{\mathrm{Re}}{3} \frac{N u}{\mu},`
)}

function _25(tex){return(
tex.block`\tag{2} 
\frac{dN}{dx} = \frac{du}{dx} - \frac{1}{\mathrm{Fr}^2}\frac{\tau_g}{u},`
)}

function _26(tex,md){return(
md`with Reynolds number Re, Froude number Fr and fiber length ${tex`L`}`
)}

function _27(tex){return(
tex.block`\mathrm{Re} = \frac{\rho_0 u_0 L}{\mu_0},`
)}

function _28(tex){return(
tex.block`\mathrm{Fr} = \frac{u_0}{\sqrt{g L}},`
)}

function _29(tex){return(
tex.block`L = \lVert r_b - r_a \rVert,`
)}

function _30(tex,md){return(
md`for domain ${tex`x \in [0,1]`} with boundary conditions ${tex`u(x=0) = u_{in}`}
and ${tex`\\`} ${tex` u(x=1) = u_{out}`}.`
)}

function _31(tex,md){return(
md`In the equations above, ${tex`\mu`} represents the dimensionless viscosity of the polymer utilized, ${tex`g`} denotes gravity, and ${tex`\tau_g`} represents the fiber direction component parallel to the gravitational force. ${tex`\rho_0`}, ${tex`u_0`}, and ${tex`\mu_0`} refer to the reference density, velocity, and viscosity of the polymer, respectively. Additionally, ${tex`u_{in}`} and ${tex`u_{out}`} correspond to the reference inlet and outlet velocities, respectively.`
)}

function _32(md){return(
md`## Learning Objective
Now, let us attempt to construct a data-driven network  to solve the aforementioned ODE system for a specific process condition, incorporating the following equation parameters:`
)}

function _33(tex,md){return(
md`* Fiber Length ${tex`L`} = 1.9599 m
* Gravity ${tex`g`} = 9.884470 ${tex`\mathrm{ms}^{-2} \\`}
* Fiber direction component parallel to gravitational force ${tex`\tau_g`} = 0.8212
* Dimensionless viscosity of the polymer ${tex`\mu`} = 101.5962
* Dimensionless inlet velocity ${tex`u_{in}`} = 0.145879
* Dimensionless outlet velocity ${tex`u_{out}`} = 8.233729
* Reference density of the polymer ${tex`\rho_0`} = 1109.067160 ${tex`\mathrm{kg}\,\mathrm{m}^{-3}\\`}
* Reference velocity ${tex`u_0`} = 1.0 ${tex`\mathrm{ms}^{-1}\\`}
* Reference viscosity ${tex`\mu_0`} = 1.0 ${tex`\mathrm{Pa}\, \mathrm{s}\\`}`
)}

function _34(tex,md){return(
md`To gather training data, we employ numerical differential equation solvers for solving the boundary value problem given by (1) and (2) yielding solutions for ${tex`u`} and ${tex`N`} on a grid ${tex`(0=x_d^1, x_d^2 \ldots x_d^{N_d}=1)`}. The network architecture consists of three hidden layers, each containing 50 neurons. The activation function chosen for both networks is the hyperbolic tangent (tanh) function. Through training, our aim is to minimize the loss by computing the mean squared error (MSE) between the predicted solution and the actual solution obtained from the numerical solvers.`
)}

function _35(tex){return(
tex.block`\tag{3} 
\mathrm{Data Loss} (Loss_{d}) = \frac{1}{N_{d}} \sum_{i=1}^{N_{d}} \frac{|u(x_d^i) - \hat{u}(x_d^i)|^2 + |N(x_d^i) - \hat{N}(x_d^i)|^2}{2}`
)}

function _36(tex,md){return(
md`Here, ${tex`N_d`} represents the total number of data points. ${tex`u(x_d^i)`} and ${tex`N(x_d^i)`} denote the numerical solutions, while ${tex`\hat{u}(x_d^i)`} and ${tex`\hat{N}(x_d^i)`} represent the predicted solutions. The subscript ${tex`d`} signifies that these values correspond to data points derived from the training set.`
)}

function _37(md){return(
md`In order to effectively train this network, it is necessary to have access to high-quality data covering the entire domain. However, obtaining such data in real-world application scenarios is often impractical. Typically, we encounter data that is sparse, noisy, and incomplete. In such cases, applying the data-driven approach can lead to undesirable outcomes. To illustrate this, consider the following example where our network is trained based on the available data within a specific domain range. Please use the slider to expand the range and thereby increase the number of data points used for training the network. From the interactive plot depicted below, it is evident that the network fails to extrapolate across the entire domain unless we possess complete data.`
)}

function _38(DOM,Plotly,data_to_plot)
{ 
  let layout = {
    width: 550,
    title: 'Solution velocity u',
    xaxis: {
      title: 'grid points',
      showgrid: true,
      zeroline: false,
    },
    yaxis: {
      title: 'non-dimensional solution',
      range: [0, 10],
      showline: false,
      showgrid: false,
    }
  }
  
  const div = DOM.element('div');
  Plotly.newPlot(div, data_to_plot, layout);
  return div;
}


function _range(Inputs){return(
Inputs.range([0.0, 1.0], {label: "grid-points range used to generate training data", step: 0.1, value:0.1})
)}

function _40(DOM,Plotly,data_to_plot_1)
{ 
  let layout = {
    width: 550,
    title: 'Solution tension N',
    xaxis: {
      title: 'grid points',
      showgrid: true,
      zeroline: false,
    },
    yaxis: {
      title: 'non-dimensional solution',
      range: [-2, 6],
      showline: false,
      showgrid: false,
    }
  }
  
  const div = DOM.element('div');
  Plotly.newPlot(div, data_to_plot_1, layout);
  return div;
}


function _41(md){return(
md`<em> In the above graphs, the green curve depicts the ground truth from the numerical simulation, i.e., the function to be approximated. Orange points are given as training data for the neural network. In the data-driven approach, the predicted function in blue is just a simplified (logical) extension of available data and cannot model variance across the entire function domain with limited training data. </em>`
)}

function _42(tex,md){return(
md`
Now, we will proceed with building a physics-informed network (PINN) to solve the same ODE system. For this, we employ the identical architecture utilized in the data-driven network. The PINN distinguishes itself from the data-driven model through the implementation of a different loss function that needs to be minimized. The general concept behind this approach is to minimize model prediction errors without relying on a predetermined ground truth. Instead, it accomplishes this by applying penalties to deviations that do not comply with the underlying differential equation. Let us consider the ODE system under examination, which is a boundary value problem with two boundary conditions ${tex`({u_{in} =u(x_b^1), u(x_b^2)=u_{out}})`}. Here, the first term of the PINN loss function aims to minimize the network prediction error for these boundary conditions, as described below.`
)}

function _43(tex){return(
tex.block`\tag{4} 
\mathrm{BoundaryLoss}(Loss_{b}) = \frac{1}{N_{b}} \sum_{i \in \{1, 2\}} |u(x_b^i) - \hat{u}(x_b^i)|^2`
)}

function _44(tex,md){return(
md`Here, ${tex`N_b`} represents the total number of boundary conditions. ${tex`u(x_b^i)`} represents the actual solution, while ${tex`\hat{u}(x_d^i)`} represents the predicted solution of velocity ${tex`u`}. The subscript ${tex`b`} indicates that these values correspond to boundary conditions.`
)}

function _45(md){return(
md`The second loss term of the PINN aims at minimizing the residual of both ODE equations (1) and (2) `
)}

function _46(tex){return(
tex.block`\tag{5} 
f_{res}(u, N, x) = \frac{du}{dx}(x) - \frac{\mathrm{Re}}{3} \frac{N(x) u(x)}{\mu},`
)}

function _47(tex){return(
tex.block`\tag{6} 
g_{res}(u, N, x) = \frac{dN}{dx}(x) - \frac{du}{dx}(x) + \frac{1}{\mathrm{Fr}^2}\frac{\tau_g}{u(x)}`
)}

function _48(tex,md){return(
md`The loss is described below on a grid ${tex`(0=x_r^1, x_r^2 \ldots x_r^{N_r}=1)`}, where ${tex`N_r`} represents the total number of residual points. The predicted solutions obtained from the network are denoted as ${tex`\hat{u}`} and ${tex`\hat{N}`}. The subscript ${tex`r`} signifies that these values correspond to residual points.`
)}

function _49(tex){return(
tex.block`\tag{7} 
\mathrm{ResidualLoss}(Loss_{r}) = \frac{1}{N_{r}} \sum_{i=1}^{N_{r}} \frac{1}{2} (f_{res}(\hat{u},\hat{N}, x_r^i)^2 + g_{res}(\hat{u}, \hat{N},x_r^i)^2)`
)}

function _50(tex,md){return(
md`In ${tex`Loss_r`}, it is required to compute the first-order derivative of the network solution at the domain points (x_r). This calculation can be accomplished using the auto-differentiation function provided by the deep-learning libraries. Below is a code snippet that demonstrates the implementation of computing the PINN loss and training using TensorFlow:

\`\`\`python
def loss_function(x_b, u_b, x_r):
    # claculate the ode residual
    x_r = tf.convert_to_tensor(x_r, dtype = tf.float32)
    with tf.GradientTape(persistent = True) as tp:
        tp.watch(x_r)
        y_pred = PINN.predict(x_r)
        u = y_pred[:, 0:1]
        N = y_pred[:, 1:]
    du = tp.gradient(u, x)
    dN = tp.gradient(N, x)
    del tp

    Re = 1.9599
    Fr = 0.2271
    mu = 101.5962
    tau_g = 0.8212

    u_residual = du - (Re / 3) * ((N * u) / mu)
    N_residual = dN - du + (1 / tf.math.square(Fr)) * (tau_g / u)
        
    loss_ode_residual =  tf.reduce_mean(tf.square(u_residual)) + tf.reduce_mean(tf.square(N_residual))
    loss_boundary = tf.reduce_mean(tf.square(x_b - u_b))
        
    loss_total = boundary_loss_weight * loss_boundary + ode_residua_loss_weight * loss_ode_residual
    return loss_total

def train_model(x_b, u_b, x_r):
    with tf.GradientTape(persistent = True) as tp:
        loss = self.loss_function(x_b, u_b, x_r)
    grad = tp.gradient(loss, PINN.trainable_params)
    optimizer.apply_gradients(zip(grad, PINN.trainable_params))
 \`\`\`

 The solution for the ODE system that is computed using PINN is visualized below.`
)}

function _51(DOM,Plotly,pinn_to_plot_u)
{ 
  let layout = {
    width: 550,
    title: 'PINN: Solution velocity u',
    xaxis: {
      title: 'grid points',
      range: [0, 1.1],
      showgrid: true,
      zeroline: false,
    },
    yaxis: {
      title: 'non-dimensional solution',
      range: [0, 9],
      showline: false,
      showgrid: false,
    }
  }
  
  const div = DOM.element('div');
  Plotly.newPlot(div, pinn_to_plot_u, layout);
  return div;
}


function _52(DOM,Plotly,pinn_to_plot_N)
{ 
  let layout = {
    width: 550,
    title: 'PINN: Solution tension N',
    xaxis: {
      title: 'grid points',
      range: [0, 1.1],
      showgrid: true,
      zeroline: false,
    },
    yaxis: {
      title: 'non-dimensional solution',
      range: [0, 6],
      showline: false,
      showgrid: false,
    }
  }
  
  const div = DOM.element('div');
  Plotly.newPlot(div, pinn_to_plot_N, layout);
  return div;
}


function _53(md){return(
md`<em> In the above graphs, the green curve depicts the ground truth from the numerical simulation, i.e., the function to be approximated. In the physics-driven approach, the predicted function in blue is able to represent the model variance across the entire function domain even without a single point of training data. </em>`
)}

function _54(cite,md){return(
md`## Training and Convergence:

We observed that the Physics-Informed Neural Network (PINN) can accurately predict the solution to an ordinary differential equation (ODE) with minimal error, even without any training data. Now, let's delve deeper into the examination of how the training process and convergence behavior of PINN compare to those of a data-driven neural network, which is trained with the data over the entire domain. To conduct this analysis, we specifically selected the Adam ${cite("c10")} and L-BFGS ${cite("c12")} optimizers, as they are commonly used in the literature concerning PINNs. Additionally, we included the RMSprop ${cite("c11")} optimizer in our analysis to explore how it performs in comparison to other optimizers. We trained our models using various learning rates and subsequently generated a visualization of the loss convergence plot for the training process, spanning 500 epochs. The drop-down menu can be used to switch between optimizers, and learning rates, and observe their respective convergence patterns.`
)}

function _opt(Inputs){return(
Inputs.select(["Adam", "Adam+LBFGS", "RMSProp+LBFGS"], {label: "Select the Optimizer"})
)}

function _lr(Inputs){return(
Inputs.select(["0.1", "0.01", "0.001"], {label: "Select the Learning Rate", value: "0.001"})
)}

function _57(DOM,Plotly,conv_plot_dd)
{ 
  let layout = {
    width: 550,
    height: 400,
    title: 'Training loss of data-driven network',
    xaxis: {
      title: 'training epochs',
      range: [0, 505],
      showgrid: true,
      zeroline: false,
    },
    yaxis: {
      title: 'log (MSE)',
      showline: false,
      showgrid: false,
    }
  }
  
  const div = DOM.element('div');
  Plotly.newPlot(div, conv_plot_dd, layout);
  return div;
}


function _58(DOM,Plotly,conv_plot_pinn)
{ 
  let layout = {
    width: 550,
    height: 400,
    title: 'Training lof physics informed network',
    xaxis: {
      title: 'training epochs',
      range: [0, 505],
      showgrid: true,
      zeroline: false,
    },
    yaxis: {
      title: 'log (MSE)',
      showline: false,
      showgrid: false,
    }
  }
  
  const div = DOM.element('div');
  Plotly.newPlot(div, conv_plot_pinn, layout);
  return div;
}


function _59(md){return(
md`<em> In the above graphs, we show the convergence behavior of the data-driven neural network (utilizing sufficient training data) and 
the physics-informed neural network (PINN) across different optimizers and learning rates. We can observe that, with a substantial volume of data, the data-driven network achieves quicker convergence compared to the PINN. Consequently, the PINN requires to be trained for longer epochs to reach the global minimum. </em>`
)}

function _60(cite,md){return(
md`Let's analyze the results we obtained. Based on the graphs, it is clear that the learning rate affects both networks similarly. Using smaller learning rates is preferable for stability, as increasing the learning rate can potentially lead to the model skipping the global minimum or getting stuck in local minima. However, there is a noticeable difference in convergence behavior between the data-driven networks and the PINN. The data-driven networks tend to converge faster, requiring fewer training epochs compared to the PINN models. To gain a deeper understanding of the convergence behavior, we plotted the loss landscape for both models. We achieved this by perturbing the neural networks in two random orthogonal directions across a specific grid and visualizing the corresponding loss ${cite("c13")}. Upon examining the loss surface, it becomes apparent that the data-driven network exhibits a steeper landscape compared to the PINN. This finding helps explain why the PINN requires more epochs to reach the global minimum.  Furthermore, it is worth noting that reducing the loss initially using a first-order optimizer like Adam or RMSprop, and then minimizing it further using a second-order optimizer like L-BFGS, leads to faster convergence compared to relying solely on the first-order optimizers.`
)}

function _61(width,DOM,Plotly,plot_loss_dd)
{    
  let layout = {
    title: 'Loss Landscape of Data-driven Network',
    autosize: false,
    width: width * 0.6,
    height: width * 0.5,
    margin: {
      l: 65,
      r: 50,
      b: 65,
      t: 90,
    },
   scene: {
		xaxis:{title: 'x', ticktext:['-1','-0.5', '0', '0.5', '1.0'], tickvals:[0, 10, 20, 30, 40]},
		yaxis:{title: 'y', ticktext:['-1','-0.5', '0', '0.5', '1.0'], tickvals:[0, 10, 20, 30, 40]},
		zaxis:{title: 'normalized network loss'},
		}
  };
  
  const div_loss_dd = DOM.element('p');
  Plotly.newPlot(div_loss_dd, plot_loss_dd, layout);
  return div_loss_dd;
}


function _62(unpack,z_data_pinn,width,DOM,Plotly)
{
  var zData = [];
  
  for (var i=0; i<40; i++) {
    zData.push(unpack(z_data_pinn,i));
  }
  
  var data = [{
    z: zData,
    type: 'surface',
    visible : true,
  }];
    
  var layout = {
    title: 'Loss Landscape of PINN',
    autosize: false,
    width: width * 0.6,
    height: width * 0.5,
    margin: {
      l: 65,
      r: 50,
      b: 65,
      t: 90,
    },
   scene: {
		xaxis:{title: 'x', ticktext:['-1','-0.5', '0', '0.5', '1.0'], tickvals:[0, 10, 20, 30, 40]},
		yaxis:{title: 'y', ticktext:['-1','-0.5', '0', '0.5', '1.0'], tickvals:[0, 10, 20, 30, 40]},
		zaxis:{title: 'normalized network loss'},
		}
  };
  
  const div_loss_pinn = DOM.element('p');
  Plotly.newPlot(div_loss_pinn, data, layout);
  return div_loss_pinn;
}


function _63(md){return(
md`<em> The above plots visualize the loss landscapes of data-driven and physics-driven neural networks respectively. We can observe from these plots that the loss landscape of the data-driven network exhibits a steeper profile in comparison to that of the physics-driven network.</em>`
)}

function _64(tex,md){return(
md`## Synergy between Data-driven and Physics-driven Networks

We have successfully constructed a PINN that accurately predicts the solution of an ODE system even without the presence of labeled data points, given fixed ODE parameters (purely physics-driven). However, in practical scenarios, it is often necessary for the model to make predictions for a range of parameters that govern the equation. Hence, we require a parameterized PINN that can be trained to handle different parameter ranges in such cases. To address this need, let's proceed with building a PINN for varying reference density ${tex`\rho_0`} within the range of (800, 1300) using the same architecture as before.

Through the utilization of the interactive visualization provided below, we can observe that the purely physics-driven PINNs (with only 'BoundaryLoss' and 'ResidualLoss' checkbox ticked) exhibit limited predictive capabilities for complex parameterized problems.
However, this limitation can be alleviated by incorporating labeled data (selecting the check box 'DataLoss'). There exists a synergistic relationship between the data-driven loss and the physics-driven losses, as they act as regularizers for each other. This implies that when used together, they can mutually enhance performance. Consequently, the results yield a hybrid PINN model that combines both data-driven and physics-driven aspects.`
)}

function _65(tex){return(
tex.block`Loss_{data+physics} = Loss_{d} + Loss_{b} + Loss_{r}`
)}

function _loss_term(Inputs){return(
Inputs.checkbox(["DataLoss", "BoundaryLoss", "ResidualLoss"], {label: "Select the loss to be used in optimization", value: ['BoundaryLoss', 'ResidualLoss']})
)}

function _data_points(Inputs){return(
Inputs.radio([6, 13, 26, 52, 130], {label: "Select the number of training data points", value: 6})
)}

function _68(unpack,ground_truth_surface_n,get_z_data,loss_term,data_points,width,DOM,Plotly)
{
  let zData = [];
  
  for (let i=0; i<100; i++) {
    zData.push(unpack(ground_truth_surface_n,i));
  }
  
  let data = [{
    z: zData,
    type: 'surface',
    colorscale: [[0, 'green'], [1, 'green']],
    opacity: 0.4,
    name : 'ground-truth',
    showscale: false,
    showlegend: true
  },
  {
    z: get_z_data(loss_term, data_points),
    type: 'surface',
    colorscale: [[0, 'red'], [1, 'red']],
    opacity: 0.4,
    name : 'prediction',
    showscale: false,
    showlegend: true
  }];
    
  let layout = {
    title: 'Solution tension N',
    autosize: false,
    width: width * 0.7,
    height: width * 0.7,
    showlegend: true,
   scene: {
		xaxis:{title: 'density (kg/m^3)', ticktext:[800, 980, 1060, 1140, 1220, 1300], tickvals:[0, 20, 40, 60, 80, 100]},
		yaxis:{title: 'grid-points', ticktext:['0.2','0.4', '0.6', '0.8', '1.0'], tickvals:[20, 40, 60, 80, 100]},
		zaxis:{title: 'solution N'}
		}
  };
  
  const div = DOM.element('div');
  Plotly.react(div, data, layout);
  return div;
}


function _69(tex,md){return(
md`<em> In the interactive surface plot graph above, we can visualize the performance of PINNs with and without data points. The checkbox enables the selection of the loss function utilized for training the model, while the radio button facilitates the adjustment of the data points employed for training (if ${tex`Loss_d`} is part of the network loss). This allows for increasing or decreasing the number of the data points as necessary.</em>`
)}

function _70(md){return(
md`Furthermore, the visualization demonstrates the existence of the synergy, up to a certain extent, which implies that when we have a sufficient amount of data that adequately represents the full variability of the problem being solved, the inclusion of physics-based constraints may no longer provide additional benefits in terms of improving performance.`
)}

function _71(md){return(
md`## Reliability Evaluation

In the preceding section, we observed the favorable accuracy achieved by a hybrid network that combines both data-driven and physics-driven approaches. Now, let's turn our attention to assessing the reliability of this network—a crucial consideration in the field of AI, as it influences the level of trust users can have in the network's predictions. We conducted evaluations to measure the network's reliability across various aspects. Specifically, we examined the performance of both the purely data-driven model and the hybrid data+physics model in the presence of noisy data, outlier data, and unseen out-of-distribution (OOD) data.`
)}

function _72(iterate_columns,ground_truth_surface_n,get_reliablity_data,radios,unpack,reliability_data,width,DOM,Plotly)
{
  
  let data = [{
    z: iterate_columns(ground_truth_surface_n, 100),
    type: 'surface',
    colorscale: [[0, 'green'], [1, 'green']],
    opacity: 0.8,
    name : 'ground-truth',
    showscale: false,
    showlegend: true
  },
  {
    z: get_reliablity_data(radios),
    type: 'surface',
    colorscale: [[0, 'red'], [1, 'red']],
    opacity: 0.4,
    name : 'prediction',
    showscale: false,
    showlegend: true
  },
  {
    x: unpack(reliability_data, 'density' + radios),
    y: unpack(reliability_data, 'grid_points' + radios),
    z: unpack(reliability_data,  'solution_N' + radios),
    name: "training-data",
    type: 'scatter3d',
    mode: 'markers',
    marker : { size : 5, color: 'black'},
    opacity: 0.4,
    showscale: false,
    showlegend: true
  }];
    
  let layout = {
    title: 'Data-driven Network',
    autosize: false,
    width: width * 0.6,
    height: width * 0.4,
   scene: {
		xaxis:{title: 'density (kg/m^3)', ticktext:[980, 1060, 1140, 1220, 1300], tickvals:[20, 40, 60, 80, 100]},
		yaxis:{title: 'grid-points', ticktext:['0.2','0.4', '0.6', '0.8', '1.0'], tickvals:[20, 40, 60, 80, 100]},
		zaxis:{title: 'solution tension N (dimless)'},
    camera: {up: {x:0, y:0, z:1}, center: {x:0, y:0, z:0}, eye: {x:2.19, y:0.1, z:0.0}}
		},
    margin: {l:0, r:0, t:30, b:0}
  };
  
  const div = DOM.element('div');
  Plotly.react(div, data, layout);
  return div;
}


function _radios(Inputs){return(
Inputs.radio(["Original Data", "Add Noise to Data", "Add Outliers", "Out of Distribution Data"], {label: "", value: "Original Data"})
)}

function _74(unpack,ground_truth_surface_n,get_reliablity_pinn,radios,reliability_data,width,DOM,Plotly)
{
  let zData = [];
  
  for (let i=0; i<100; i++) {
    zData.push(unpack(ground_truth_surface_n,i));
  }
  
  let data = [{
    z: zData,
    type: 'surface',
    colorscale: [[0, 'green'], [1, 'green']],
    opacity: 0.8,
    name : 'ground-truth',
    showscale: false,
    showlegend: true
  },
  {
    z: get_reliablity_pinn(radios),
    type: 'surface',
    colorscale: [[0, 'red'], [1, 'red']],
    opacity: 0.4,
    name : 'prediction',
    showscale: false,
    showlegend: true
  },
  {
    x: unpack(reliability_data, 'density' + radios),
    y: unpack(reliability_data, 'grid_points' + radios),
    z: unpack(reliability_data,  'solution_N' + radios),
    name: "training-data",
    type: 'scatter3d',
    mode: 'markers',
    marker : { size : 5, color: 'black'},
    opacity: 0.4,
    showscale: false,showlegend: true
  }];
    
  let layout = {
    title: 'Data+Physics-driven Network',
    autosize: false,
    width: width * 0.6,
    height: width * 0.4,
   scene: {
		xaxis:{title: 'density (kg/m^3)', ticktext:[980, 1060, 1140, 1220, 1300], tickvals:[20, 40, 60, 80, 100]},
		yaxis:{title: 'grid-points', ticktext:['0.2','0.4', '0.6', '0.8', '1.0'], tickvals:[20, 40, 60, 80, 100]},
		zaxis:{title: 'solution tension N (dimless)'},
    camera: {up: {x:0, y:0, z:1}, center: {x:0, y:0, z:0}, eye: {x:2.19, y:0.1, z:0.0}}
		},
    margin: {l:0, r:0, t:30, b:0}
  };
  
  const div = DOM.element('div');
  Plotly.react(div, data, layout);
  return div;
}


function _75(md){return(
md`<em> The graphs presented above display the performance of both a purely data-driven network and a hybrid data+physics-driven network across various reliability dimensions. To explore these distinct reliability aspects, please utilize the provided radio buttons. By switching between the aspects, you can view the corresponding performance of the models in each scenario. </em>`
)}

function _76(md){return(
md`From the observations, it is evident that the hybrid PINN exhibits better performance in handling noisy data, demonstrates robustness against outliers, and performs well on out-of-distribution (OOD) data when compared to the purely data-driven network. This indicates that the trustworthiness of the hybrid PINN is not solely dependent on the reliability of the training data. By incorporating the governing physics of the problem, the hybrid model gains an additional level of trustability, enabling it to handle various practical scenarios more effectively. `
)}

function _77(md){return(
md`## Conclusion

In summary, we have presented the concept of informed machine learning, which incorporates external knowledge into machine learning models. As part of this framework, we introduced physics-driven neural networks as a complementary approach to traditional data-driven neural networks. Through an analysis of various aspects, focusing on the application of melt spinning, we have explored the synergies between these approaches. We have highlighted the significance of integrating physics principles into machine learning models, as it enhances both their reliability and overall performance.`
)}

function _78(md){return(
md`## References`
)}

function _79(bibliography){return(
bibliography
)}

function _80(md){return(
md`## Utils and data`
)}

function _Plotly(require){return(
require("https://cdn.plot.ly/plotly-latest.min.js")
)}

function _csv(Plotly){return(
async function csv(url) {
  return Plotly.d3.csv.parse((await (await fetch(url)).text()));
}
)}

function _unpack(){return(
function unpack(rows, key) {
  return rows.map(function(row) { 
    return row[key]; 
  });
}
)}

function _get_loss_string(){return(
function get_loss_string(loss){
  if (loss.length > 0)
  {
    return loss.reduce((prev, next) => `${prev}_${next}`)
  }
  return loss
}
)}

function _get_substring(){return(
function get_substring(str) {
  return str.substring(str.indexOf("m") + 1)
}
)}

function _is_legend(){return(
function is_legend(str) {
  if (str == 'Adam')
  {
    return false
  }
  else
  {
    return true
  }
}
)}

function _get_z_data(get_loss_string,unpack,D20,D40,D60,D80,DB100,B100,P80,DB20,DB40,DB60,DB80,DP20,DP40,DP60,DP80,DP100,BP100,DBP20,DBP40,DBP60,DBP80,DBP100){return(
function get_z_data(loss_term, dp) {
  let zData = [];
  let loss_data_str = get_loss_string(loss_term)
  for (let i=0; i<100; i++) {
    if (loss_data_str == 'DataLoss')
    {
      switch (dp) {
        case 6:
          zData.push(unpack(D20,i));
          break;
        case 13:
          zData.push(unpack(D40,i));
          break;
        case 26:
          zData.push(unpack(D60,i));
          break;
        case 52:
          zData.push(unpack(D80,i));
          break;
        case 130:
          zData.push(unpack(DB100,i));
          break;
      }
    }
    else if (loss_data_str == 'BoundaryLoss')
    {
      zData.push(unpack(B100,i));
    }
    else if (loss_data_str == 'ResidualLoss')
    {
      zData.push(unpack(P80,i));
    }
    else if (loss_data_str == 'DataLoss_BoundaryLoss')
    {
      switch (dp) {
        case 6:
          zData.push(unpack(DB20,i));
          break;
        case 13:
          zData.push(unpack(DB40,i));
          break;
        case 26:
          zData.push(unpack(DB60,i));
          break;
        case 52:
          zData.push(unpack(DB80,i));
          break;
        case 130:
          zData.push(unpack(DB100,i));
          break;
      }
    }
    else if (loss_data_str == 'DataLoss_ResidualLoss')
    {
      switch (dp) {
        case 6:
          zData.push(unpack(DP20,i));
          break;
        case 13:
          zData.push(unpack(DP40,i));
          break;
        case 26:
          zData.push(unpack(DP60,i));
          break;
        case 52:
          zData.push(unpack(DP80,i));
          break;
        case 130:
          zData.push(unpack(DP100,i));
          break;
      }
    }
    else if (loss_data_str == 'BoundaryLoss_ResidualLoss')
    {
      zData.push(unpack(BP100,i));
    }
    else if (loss_data_str == 'DataLoss_BoundaryLoss_ResidualLoss')
    {
      switch (dp) {
        case 6:
          zData.push(unpack(DBP20,i));
          break;
        case 13:
          zData.push(unpack(DBP40,i));
          break;
        case 26:
          zData.push(unpack(DBP60,i));
          break;
        case 52:
          zData.push(unpack(DBP80,i));
          break;
        case 130:
          zData.push(unpack(DBP100,i));
          break;
      }
    }
    else
    {
      // empty string, need not to be updated
    }
  }
  return zData;
}
)}

function _get_reliablity_data(iterate_columns,reliability_dd_orig,reliability_dd_noise,reliability_dd_outlier,reliability_dd_ood){return(
function get_reliablity_data(radio_button){
  let data = [];
  if (radio_button ==  'Original Data')
  {
    data = iterate_columns(reliability_dd_orig, 100)
  }
  else if (radio_button ==  'Add Noise to Data')
  {
    data = iterate_columns(reliability_dd_noise, 100)
  }
  else if (radio_button ==  'Add Outliers')
  {
    data = iterate_columns(reliability_dd_outlier, 100)
  }
  else
  {
    data = iterate_columns(reliability_dd_ood, 100)
  }
  return data;
}
)}

function _get_reliablity_pinn(iterate_columns,reliability_pinn,reliability_pinn_noise,reliability_pinn_outlier,reliability_pinn_ood){return(
function get_reliablity_pinn(radio_button){
  let data = [];
  if (radio_button ==  'Original Data')
  {
    data = iterate_columns(reliability_pinn, 100)
  }
  else if (radio_button ==  'Add Noise to Data')
  {
    data = iterate_columns(reliability_pinn_noise, 100)
  }
  else if (radio_button ==  'Add Outliers')
  {
    data = iterate_columns(reliability_pinn_outlier, 100)
  }
  else
  {
    data = iterate_columns(reliability_pinn_ood, 100)
  }
  return data;
}
)}

function _iterate_columns(unpack){return(
function iterate_columns(data, index) {
  let zData = [];
  for (var i=0; i<index; i++) {
    zData.push(unpack(data,i));
  }
  return zData
}
)}

function _iterate_rows(){return(
function iterate_rows(data, min, max) {
  let zData = [];
  for (var i=min; i<max; i++) {
    zData.push(data[i]);
  }
  return zData
}
)}

function _bibliography(bib){return(
bib({
  c1: `Von Rueden, Laura, et al. Informed Machine Learning–A taxonomy and survey of integrating prior knowledge into learning systems. *IEEE Transactions on Knowledge and Data Engineering* **35.1**, 614-633 (2021).`,
  c2: `Raissi, M., Perdikaris, P., Karniadakis, G.E. Physics informed deep learning (part I): Data-driven solutions of nonlinear partial differential equations. *arXiv preprint arXiv:1711.10561* (2017).`,
  c3: `Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em. Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations. *arXiv:1711.10566 (2017)* (2017).`,
  c4: `Baty, Hubert. Solving stiff ordinary differential equations using physics informed neural networks (PINNs): simple recipes to improve training of vanilla-PINNs. *arXiv preprint arXiv:2304.08289* (2023).`,
  c5: `Pang, Guofei and Lu, Lu and Karniadakis, George Em. fPINNs: Fractional physics-informed neural networks. *SIAM Journal on Scientific Computing* **41.4**, A2603--A2626 (2019).`,
  c6: `Yuan, Lei and Ni, Yi-Qing and Deng, Xiang-Yun and Hao, Shuo. A-PINN: Auxiliary physics informed neural networks for forward and inverse problems of nonlinear integro-differential equations. *Journal of Computational Physics*, Elsevier, **462**, 111260 (2022).`,
  c7: `Shin, Hyomin and Choi, Minseok. Physics-informed variational inference for uncertainty quantification of stochastic differential equations. *Journal of Computational Physics*, Elsevier **487**, 112183 (2023).`,
  c8: `Wegener, Raimund and Marheineke, Nicole and Hietel, Dietmar. Currents in industrial mathematics: from concepts to research to education. *Springer*, 103–162 (2015).`,
  c9: `Ettmüller, Manuel and Arne, Walter and Marheineke, Nicole and Wegener, Raimund. Product integration method for the simulation of radial effects in fiber melt spinning of semi-crystalline polymers. *PAMM*, Wiley Online Library **22.1** (2023).`,
  c10: `Kingma, Diederik P and Ba, Jimmy. Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980* (2014).`,
  c11: `Tieleman, Tijmen and Hinton, Geoffrey and others. Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude. *COURSERA: Neural networks for machine learning* **4.2**, 26--31 (2012).`,
  c12: `Nocedal, Jorge. Updating quasi-Newton matrices with limited storage. *Mathematics of computation* **35.151**, 773--782 (1980).`,
  c13: `Li, Hao and Xu, Zheng and Taylor, Gavin and Studer, Christoph and Goldstein, Tom. Visualizing the loss landscape of neural nets. *Advances in neural information processing systems* **31** (2018).`,
  c14: `Victor, Viny Saajan and Schmeißer, Andre and Leitte, Heike and Gramsch, Simone. Visual parameter space analysis for optimizing the quality of industrial nonwovens. *IEEE Computer Graphics and Applications* **42.2**, 56--67 (2022).`
})
)}

function _cite($0){return(
$0
)}

function _data_to_plot(ground_truth_single_eq,range,unpack,dd_single_eq_obs){return(
[
    {
      x: ground_truth_single_eq.map(x=>x.grid_points),
      y: ground_truth_single_eq.map(x=>x.solution_u),
      mode: 'lines',
      name: "ground truth",
      line: { color: 'rgba(0, 255, 0, 0.4)', width: 7}
    },
    {
      x: ground_truth_single_eq.filter(x => x.grid_points <= range).map(x=>x.grid_points),
      y: ground_truth_single_eq.filter(x => x.grid_points <= range).map(x=>x.solution_u),
      mode: 'markers',
      name: "training-data",
    },
    {
      x: unpack(dd_single_eq_obs, 'grid points'),
      y: unpack(dd_single_eq_obs, range + 'u'),
      mode: 'lines',
      name: "data-driven network",
      line: { color: 'blue'}
    }
  ]
)}

function _data_to_plot_1(ground_truth_single_eq,range,unpack,dd_single_eq_obs){return(
[
    {
      x: ground_truth_single_eq.map(x=>x.grid_points),
      y: ground_truth_single_eq.map(x=>x.solution_N),
      mode: 'lines',
      name: "ground truth",
      line: { color: 'rgba(0, 255, 0, 0.4)', width: 7}
    },
    {
      x: ground_truth_single_eq.filter(x => x.grid_points <= range).map(x=>x.grid_points),
      y: ground_truth_single_eq.filter(x => x.grid_points <= range).map(x=>x.solution_N),
      mode: 'markers',
      name: "training-data",
    },
    {
      x: unpack(dd_single_eq_obs, 'grid points'),
      y: unpack(dd_single_eq_obs, range + 'N'),
      mode: 'lines',
      name: "data-driven network",
      line: { color: 'blue'}
    }
  ]
)}

function _pinn_to_plot_u(ground_truth_single_eq,unpack,pinn_single_eq){return(
[
    {
      x: ground_truth_single_eq.map(x=>x.grid_points),
      y: ground_truth_single_eq.map(x=>x.solution_u),
      mode: 'lines',
      name: "ground truth",
      line: { color: 'rgba(0, 255, 0, 0.4)', width: 7}
    },
    {
      x: unpack(pinn_single_eq, 'grid_points'),
      y: unpack(pinn_single_eq, 'u(x)'),
      mode: 'lines',
      name: "PINN",
      line: { color: 'blue'}
    }
  ]
)}

function _pinn_to_plot_N(ground_truth_single_eq,unpack,pinn_single_eq){return(
[
    {
      x: ground_truth_single_eq.map(x=>x.grid_points),
      y: ground_truth_single_eq.map(x=>x.solution_N),
      mode: 'lines',
      name: "ground truth",
      line: { color: 'rgba(0, 255, 0, 0.4)', width: 7}
    },
    {
      x: unpack(pinn_single_eq, 'grid_points'),
      y: unpack(pinn_single_eq, 'N(x)'),
      mode: 'lines',
      name: "PINN",
      line: { color: 'blue'}
    }
  ]
)}

function _conv_plot_dd(unpack,data_loss_adam_rms,opt,lr,data_lbfgs_end,get_substring,is_legend){return(
[
    {
      x: unpack(data_loss_adam_rms, 'epoch'),
      y: unpack(data_loss_adam_rms, opt + lr),
      mode: 'lines',
      name: "Adam",
    },
    {
      x: unpack(data_lbfgs_end, 'epoch'),
      y: unpack(data_lbfgs_end, opt + lr),
      mode: 'lines',
      name: get_substring(opt),
      showlegend: is_legend(opt)
    }
  ]
)}

function _conv_plot_pinn(unpack,pinn_loss_adam_rms,opt,lr,pinn_lbfgs_end,get_substring,is_legend){return(
[
    {
      x: unpack(pinn_loss_adam_rms, 'epoch'),
      y: unpack(pinn_loss_adam_rms, opt + lr),
      mode: 'lines',
      name: "Adam",
    },
    {
      x: unpack(pinn_lbfgs_end, 'epoch'),
      y: unpack(pinn_lbfgs_end, opt + lr),
      mode: 'lines',
      name: get_substring(opt),
      showlegend: is_legend(opt)
    }
  ]
)}

function _plot_loss_dd(iterate_columns,z_data_data){return(
[
  {
    z: iterate_columns(z_data_data, 40),
    type: 'surface'
  }
]
)}

async function _numpy(pyodide,py)
{
  await pyodide.loadPackage("numpy");
  return py`import numpy
numpy`;
}


function _ground_truth_surface_u(__query,FileAttachment,invalidation){return(
__query(FileAttachment("ground_truth_surface_u@3.csv"),{from:{table:"ground_truth_surface_u"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _ground_truth_surface_n(__query,FileAttachment,invalidation){return(
__query(FileAttachment("gt_surface_N_obs.csv"),{from:{table:"gt_surface_N_obs"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _pinn_lbfgs_end(__query,FileAttachment,invalidation){return(
__query(FileAttachment("lbfgs_end_pinn.csv"),{from:{table:"lbfgs_end_pinn"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _pinn_loss_adam_rms(__query,FileAttachment,invalidation){return(
__query(FileAttachment("pinn_loss_adam_rms.csv"),{from:{table:"pinn_loss_adam_rms"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _ground_truth_single_eq(__query,FileAttachment,invalidation){return(
__query(FileAttachment("ground_truth_single_eq.csv"),{from:{table:"ground_truth_single_eq"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _dd_single_eq_obs(__query,FileAttachment,invalidation){return(
__query(FileAttachment("dd_single_eq_obs@1.csv"),{from:{table:"dd_single_eq_obs"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _pinn_single_eq(__query,FileAttachment,invalidation){return(
__query(FileAttachment("pinn_single_eq.csv"),{from:{table:"pinn_single_eq"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _data_loss_adam_rms(__query,FileAttachment,invalidation){return(
__query(FileAttachment("data_loss_adam_rms.csv"),{from:{table:"data_loss_adam_rms"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _data_lbfgs_end(__query,FileAttachment,invalidation){return(
__query(FileAttachment("lbfgs_end_data.csv"),{from:{table:"lbfgs_end_data"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _z_data_data(__query,FileAttachment,invalidation){return(
__query(FileAttachment("z_data_data@1.csv"),{from:{table:"z_data_data"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _z_data_pinn(__query,FileAttachment,invalidation){return(
__query(FileAttachment("z_data_pinn@1.csv"),{from:{table:"z_data_pinn"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _D20(__query,FileAttachment,invalidation){return(
__query(FileAttachment("0.5data_surface.csv"),{from:{table:"0.5data_surface"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _DB20(__query,FileAttachment,invalidation){return(
__query(FileAttachment("0@1.5data_bc_surface.csv"),{from:{table:"0.5data_bc_surface"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _DP20(__query,FileAttachment,invalidation){return(
__query(FileAttachment("0@1.5data_pinn_surface.csv"),{from:{table:"0.5data_pinn_surface"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _DBP20(__query,FileAttachment,invalidation){return(
__query(FileAttachment("0@1.5data_bc_pinn_surface.csv"),{from:{table:"0.5data_bc_pinn_surface"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _D40(__query,FileAttachment,invalidation){return(
__query(FileAttachment("0.6data_surface.csv"),{from:{table:"0.6data_surface"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _DB40(__query,FileAttachment,invalidation){return(
__query(FileAttachment("0@1.6data_bc_surface.csv"),{from:{table:"0.6data_bc_surface"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _DP40(__query,FileAttachment,invalidation){return(
__query(FileAttachment("0@1.6data_pinn_surface.csv"),{from:{table:"0.6data_pinn_surface"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _DBP40(__query,FileAttachment,invalidation){return(
__query(FileAttachment("0@1.6data_bc_pinn_surface.csv"),{from:{table:"0.6data_bc_pinn_surface"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _D60(__query,FileAttachment,invalidation){return(
__query(FileAttachment("0.7data_surface.csv"),{from:{table:"0.7data_surface"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _DB60(__query,FileAttachment,invalidation){return(
__query(FileAttachment("0@1.7data_bc_surface.csv"),{from:{table:"0.7data_bc_surface"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _DP60(__query,FileAttachment,invalidation){return(
__query(FileAttachment("0@1.7data_pinn_surface.csv"),{from:{table:"0.7data_pinn_surface"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _DBP60(__query,FileAttachment,invalidation){return(
__query(FileAttachment("0@1.7data_bc_pinn_surface.csv"),{from:{table:"0.7data_bc_pinn_surface"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _D80(__query,FileAttachment,invalidation){return(
__query(FileAttachment("0.8data_surface.csv"),{from:{table:"0.8data_surface"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _P80(__query,FileAttachment,invalidation){return(
__query(FileAttachment("0.8pinn_surface.csv"),{from:{table:"0.8pinn_surface"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _DB80(__query,FileAttachment,invalidation){return(
__query(FileAttachment("0@1.8data_bc_surface.csv"),{from:{table:"0.8data_bc_surface"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _DP80(__query,FileAttachment,invalidation){return(
__query(FileAttachment("0@1.8data_pinn_surface.csv"),{from:{table:"0.8data_pinn_surface"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _DBP80(__query,FileAttachment,invalidation){return(
__query(FileAttachment("0@1.8data_bc_pinn_surface.csv"),{from:{table:"0.8data_bc_pinn_surface"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _D100(__query,FileAttachment,invalidation){return(
__query(FileAttachment("0@2.9data_surface.csv"),{from:{table:"0.9data_surface"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _B100(__query,FileAttachment,invalidation){return(
__query(FileAttachment("0.9bc_surface.csv"),{from:{table:"0.9bc_surface"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _DB100(__query,FileAttachment,invalidation){return(
__query(FileAttachment("0@1.9data_bc_surface.csv"),{from:{table:"0.9data_bc_surface"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _BP100(__query,FileAttachment,invalidation){return(
__query(FileAttachment("0.9bc_pinn_surface.csv"),{from:{table:"0.9bc_pinn_surface"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _DP100(__query,FileAttachment,invalidation){return(
__query(FileAttachment("0@1.9data_pinn_surface.csv"),{from:{table:"0.9data_pinn_surface"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _DBP100(__query,FileAttachment,invalidation){return(
__query(FileAttachment("0@1.9data_bc_pinn_surface.csv"),{from:{table:"0.9data_bc_pinn_surface"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _reliability_data(__query,FileAttachment,invalidation){return(
__query(FileAttachment("reliability_data_dd@1.csv"),{from:{table:"reliability_data_dd"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _reliability_dd_orig(__query,FileAttachment,invalidation){return(
__query(FileAttachment("reliability_dd_orig.csv"),{from:{table:"reliability_dd_orig"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _reliability_dd_noise(__query,FileAttachment,invalidation){return(
__query(FileAttachment("reliability_dd_noise.csv"),{from:{table:"reliability_dd_noise"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _reliability_dd_outlier(__query,FileAttachment,invalidation){return(
__query(FileAttachment("reliability_dd_outlier.csv"),{from:{table:"reliability_dd_outlier"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _reliability_dd_ood(__query,FileAttachment,invalidation){return(
__query(FileAttachment("reliability_dd_ood.csv"),{from:{table:"reliability_dd_ood"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _reliability_pinn(__query,FileAttachment,invalidation){return(
__query(FileAttachment("reliability_pinn.csv"),{from:{table:"reliability_pinn"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _reliability_pinn_noise(__query,FileAttachment,invalidation){return(
__query(FileAttachment("reliability_pinn_noise.csv"),{from:{table:"reliability_pinn_noise"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _reliability_pinn_outlier(__query,FileAttachment,invalidation){return(
__query(FileAttachment("reliability_pinn_outlier.csv"),{from:{table:"reliability_pinn_outlier"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

function _reliability_pinn_ood(__query,FileAttachment,invalidation){return(
__query(FileAttachment("reliability_pinn_ood.csv"),{from:{table:"reliability_pinn_ood"},sort:[],slice:{to:null,from:null},filter:[],select:{columns:null}},invalidation)
)}

export default function define(runtime, observer) {
  const main = runtime.module();
  function toString() { return this.url; }
  const fileAttachments = new Map([
    ["Sankey.png", {url: new URL("./files/9fb1217224fe63a099f03eef06c84b4e05d15b77ee7cb0efc807d6dd2c3e35b7933b2b6c1bd722d5ff265fe2cea937a80f1c237f5ca89a2d4e2b2149bad365c5.png", import.meta.url), mimeType: "image/png", toString}],
    ["Sankey_PINN.png", {url: new URL("./files/7df8e4c44dc86747f44a4234bcb884015ec8ef923bffab8323a19905c467b6366788a695ddac6f408493bd7fcf9d28760f4d0cead5bc011123d8bb3e87540657.png", import.meta.url), mimeType: "image/png", toString}],
    ["ground_truth_single_eq.csv", {url: new URL("./files/8eadf3cda88679470d541f12ff40ad01b4ac1527a716a97e0a8a0a13b3883e7421cf47e22a78b11900514c06e082c0a73fff1559487b011c2220b5a149a8f4cb.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["ground_truth_surface_u@3.csv", {url: new URL("./files/bbf1471b3683175cd7bcca1c513c4dca20419f55a455a24010bd09eaf1319e47b4522cff50668b728796c21fb3c04454fb07059858db38ac5b33279254aec3f5.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["0.8data_surface.csv", {url: new URL("./files/06b97888bcc87e2fa9255e283ad4416adf7cc9b551138a249e06ffd5f1fdf29edaa7250697e528c6446d973ca68693891a3a8fcc082771f4ae964ebb6e3e6caa.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["0.6data_surface.csv", {url: new URL("./files/1c0bdfc7ce2cd7b71498758cf72095d20e3775affcef50dabe2e70ee15251e7fd7140f75c289e0ee525dda1821c180ee876911ab8ab3fe9b35d71713688a179c.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["0.9bc_surface.csv", {url: new URL("./files/a3cbfa087dd7dee074dab3689ecc6216a0c5091b280ca20fbd18655bb36f2c8f3fb0d07f1e37835a387867c6b772756bbe9aea4af552cc2c123bd2adb9b2f1de.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["0.7data_surface.csv", {url: new URL("./files/4bcb2eecb297ab89825f96fe9b9924b4868430cec51bc9caf0bdc24fbc52d58793d63920f95560e95a0e97f13232c0e123bb7fb2799c6c658064c6394b4f8b33.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["0.9bc_pinn_surface.csv", {url: new URL("./files/7b1945e6916090adfe78d293cac8b6759ba4e93c9247f69354282be80d0e6953f68e507f5f0e67182747fa3bc8798faf0cfa7e41a22c142990762966881125ce.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["0.8pinn_surface.csv", {url: new URL("./files/81976221c8ace022f986040d7df4a38c554421c0f3abd3f142cf67adbb29c5fe6ad11a373e44e7391650fb4a3e713f630c86d6fd504893d7ada2116ab6aa82cb.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["0.5data_surface.csv", {url: new URL("./files/914ef326b6c34fb1b28c8b45381476dcb770a2f783588e5f4d7161434889857efbc7db903c2bbad9555a236350259c44b84b2b902a3258c74445de99abafa58b.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["gt_surface_N_obs.csv", {url: new URL("./files/1d05b95eca7831277e99d9332c9b2b6f486b4f001d5a653accfeab817603a370218ac83e0afe2d9871bc568c1ecf0b05464e14143ed59c44eb6fce25f9ff86f6.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["reliability_data_dd@1.csv", {url: new URL("./files/5787f0653adf69cdeabb9583b3fefcc3249f8eed85df9ce9857a86a9103eae9e9ea935e9b8dd1ca5a0839faa6e4e1731704a9c9079e5c46c5d7d714d1d768d77.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["reliability_dd_orig.csv", {url: new URL("./files/d3f811a79a44b83fb01abff8b6bc4301eadbea8ecbb86460361c5d7628d6ff3c1679b1fa1315b3f989945c621358c54c7af4a1c3076c15a2c7d9b304ed381ec3.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["reliability_dd_ood.csv", {url: new URL("./files/ff0b3a3e4d3a01cbf44d9e9c8a70cf1eaf7bf05ba1f0e870a88a215147a04e60d7591847423a7e9334ab92820dd50a865221eab5eb98b7bbe21a536959d238ac.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["reliability_dd_noise.csv", {url: new URL("./files/529f07c80f98d7ef2900b7e0809e4144b69e9650400f22f2191d61047b9aea57a64343f10c60ec397caa72a133536ecd751d978b53011a690b0ecc160a02550f.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["reliability_dd_outlier.csv", {url: new URL("./files/976a81646ff762b19874e2a2c577edf9d572d0a372e467b3f88eb771e3f97a278ae7300d599520b1a71474c5bbe2e07b9b67a4928ec58adb0007e7de495e000f.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["reliability_pinn.csv", {url: new URL("./files/d91b5548ca07eea98c7ec4400ac5bd8ba980c09d16c288e2a24581f9451094a0eb1adadf3cb137621a67ef759eeda811b24be42d3e2ee526b7b3762081ca6d57.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["reliability_pinn_outlier.csv", {url: new URL("./files/9f27c94cbe6c282bcc63480aa43c80f43e8908767a7a7f380aff398c9d19031bd8921b1bc840232c0d2fcb199fc5df5e7987de16488c0a6adc958dd836e356a2.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["reliability_pinn_ood.csv", {url: new URL("./files/121ad4e970073863612b4211a0f145dd5966e0ae42206919813f8bfb4fa5b35d941411cd51991d7723e6f3649f0779c6bd009bb96af4abdca3ccb52b54f3b1b8.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["reliability_pinn_noise.csv", {url: new URL("./files/af73a839d9bd3d422d6b733befdb69b70f79ebb28b50adf483672dc54b2a217a0bc5fc4b4d211cf8735a1767f441d84f00f65b93ef27677a1c7f94bc2fd7eece.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["pinn_loss_adam_rms.csv", {url: new URL("./files/5cb9fc4e5356e4e6484ef954b50be2c58e962b96f6556de7e26dd42ab486e1a693e0d43826318ab187bb479aad857cb22d00b644c2e696752689a89090a940a7.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["data_loss_adam_rms.csv", {url: new URL("./files/9d796360e20de60a7467dd03375b4762d13dac9e24f33a5810e9c8949e628390a17a805fdda62474daf5a4780756d30836f01c38de7622579f67c2ea24d8b423.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["lbfgs_end_pinn.csv", {url: new URL("./files/7045416cf21e98b152bd2e0c08aac1eb5ce0e83a0806eb2db1b55c2d5a8bf00a22e21649c671cc7f1cf6babf7a1afbc31e545ab32b37296fb40252a7a006886d.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["lbfgs_end_data.csv", {url: new URL("./files/7edfefaa7930b8911f43bbaafd1bbdf6eda0da0c308815063d08ee0f582a72595c9d0a0ded98619f6c49c65611ce2d8e4dd4195dd9c3d837bbc4152fed6edbe8.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["pinn_single_eq.csv", {url: new URL("./files/bbb084ebe675027b61b60c86f8b5f47d621040a940743430cade46e1c94acd3e1d278ea7fa99bc2d8b1ba0160f6f4ce2506365e4bc243dd881ddd13afc826e3e.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["0@1.5data_pinn_surface.csv", {url: new URL("./files/cb4f20826308a7ef2438bb7dfa180b8464950fec353708194723cb7852107d255eccde32f79dddebc05118a3375b443681f026548473b3656f6f99652831801b.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["0@1.5data_bc_surface.csv", {url: new URL("./files/d73df7868e7e6877316b4c51ebb5f16ead3d15ae41de6985ed43b3c2f459e4813ef13a989e07c7fc4c74d0f4b89a227f99b52a05889dfea00819cbf71327968c.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["0@1.5data_bc_pinn_surface.csv", {url: new URL("./files/f063e9f43b20269787e8514b3988da69f97522dad4cb9d8987f3c7b273ee872bb8cf7a92af34b1535ecf363734219d8ebfbc7c1c677530ba26c648923c125500.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["0@1.6data_bc_pinn_surface.csv", {url: new URL("./files/ee8b5326ba2ea82da49a8ca2dc1d7c2455bb35962ffea7cc34d983d4468aeab54ce2fa1f0599b6e77973b41b7724da5d91197bffbb52cccdabc90d6d4f55c932.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["0@1.6data_bc_surface.csv", {url: new URL("./files/8682bdaa85140afe4d71bfaac68101ae9387753ef81a3a50f7cbde05ca35ad2f29804c148e1f1aee13cc0d1905d19b07291608f66dd8fe3fecd7820710f8bec6.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["0@1.6data_pinn_surface.csv", {url: new URL("./files/b529e79b0c8e34b5338ce3adb3057d9928481efb61eb8dc4e0066ba72c68a040ea5a75d6e299a062b71a8bc4873ce567c564e338e026886d8cbc7a106f8bf6ca.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["0@1.7data_bc_pinn_surface.csv", {url: new URL("./files/8f3d4d8b6946ab490f261010e3f9caeb1b302c7d7255eedd16aba26d83f72ddd3462fddf7d16def8c65415e28e29abea24907c043229878166ddc8f8525ec834.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["0@1.7data_bc_surface.csv", {url: new URL("./files/deb9fae0246ddbc8b8e9a19e298963d7081f4772119f9e3e4d57d2394ebe34c6507abf2394a43ddaf21a3824a0759ba77a65ba382c0af10f8a06741d190f9e76.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["0@1.7data_pinn_surface.csv", {url: new URL("./files/247e20110442dddb27c648588445ce5e4bb5e778b3b484b201411d997a033cd7acdfe80aaebdaa8343dde152667e953905673f44b1524bdd352d9308f006cbca.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["0@1.8data_bc_pinn_surface.csv", {url: new URL("./files/3bf82860b9f24ec03bdbfee3abfb592a95d251755fb03ac0cb2ebc83fc1ed5e7e70e01f123b2f2672de0d0571d9b3cbc1d063cd4ce71212dacba545c7fc7f09f.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["0@1.8data_bc_surface.csv", {url: new URL("./files/2809b6631a99106af541a25682162ded14051580568bd5cc49448e275c9ef0f1450f0fd10bcc2140c917e9ac2a028f54f1536e050c561d8719b0e836d9c125cb.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["0@1.8data_pinn_surface.csv", {url: new URL("./files/930222522af59512c7b3b623a7abec2b1a0b53fd6941598e9590202e453df260b2eeefb7783d6763b58d47fab0d91fdbe0dbb2b72b53897d2682fca36870101c.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["0@1.9data_bc_pinn_surface.csv", {url: new URL("./files/36c6be1aa3c56f9c0793d5a68a548eac202caaa23e14268a41d753e6f4de9b26d4235c17e1bdb15a7ba7b24005cb4b24a8f62c590b84d7a4a118ea5826103031.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["0@1.9data_bc_surface.csv", {url: new URL("./files/863c20c669c9ca4f0b0cda84be2d0cf0c9c9d8fab44dd15e64391f517c8d10f35596f12cca6c7357dd2b3de4da5909ab4d8150b59421ec3ff3b792a41feff8c4.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["0@1.9data_pinn_surface.csv", {url: new URL("./files/fe1a0d7f271269569773a3eed53c9969109298024a09d4d7e8195aa907c7afe310e9b8c2010f2a18791a988e04f1a4c436d6f10ccbbb5809338063d55f3cc8ad.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["0@2.9data_surface.csv", {url: new URL("./files/a2ade9cb7eb1058cf8dfed7222fd7957e35084e1c4ecb8bf2f666665e4501d4501586f6882ba11896ea0d0b37d917bcbcf51d820d7d8f9809eed17e458d93a70.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["process_melt_spinning_simple-1.png", {url: new URL("./files/63c1b796a3cc3869e2ac514ffffb5e0b0c2a9d86782600313e1cf0408fa9af2f4d91e3883da891c8d60cabcf9a531c44d3636678a78c20e5f5b3c09db7f6ecda.png", import.meta.url), mimeType: "image/png", toString}],
    ["IML_rev.png", {url: new URL("./files/073803c86a970627529e49fc4fac3c0a3817139c2cc73e4d518dc25356fa829d7d5fab555c7f895dfcce149be9b99f0ce6a6ce6ef8ae08e88df1965e975f17f8.png", import.meta.url), mimeType: "image/png", toString}],
    ["DD_ML_rev.png", {url: new URL("./files/8ced4afcd4428681f0abfb747a9553b4871ac89862dc98f59499c27a84eabdd91bb1a63e67afa86d7b53f0565f6f4203de129f6b68befac52fbf7cd8a58d091b.png", import.meta.url), mimeType: "image/png", toString}],
    ["z_data_data@1.csv", {url: new URL("./files/bec1ee0efc7c4f644f100954268bc14e82818cd4ca01847835d9823c3f9ae7ec0f66e41d7c1ffe2ec7f7c50a10f2662df185e593909ad6684ebb404a8e0d7e62.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["z_data_pinn@1.csv", {url: new URL("./files/136a872f5880c0de548c090e2ee2c67aac560aac0bc07299ada9fea3473a53716f09fe0edff2dd25a8a3f983426e9d15cefd801890cb773cfe41431764c621b9.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["dd_single_eq_obs@1.csv", {url: new URL("./files/6698010cd9477997cc540a35a4ef76707199342409718c6577ea543392ad86e91e79cf723f9ac2d0690b44454b6493cb05b070266ada753f562ec1cf4eeaef59.csv", import.meta.url), mimeType: "text/csv", toString}]
  ]);
  main.builtin("FileAttachment", runtime.fileAttachments(name => fileAttachments.get(name)));
  main.variable(observer()).define(["md"], _1);
  main.variable(observer()).define(["md"], _2);
  main.variable(observer()).define(["md"], _3);
  main.variable(observer()).define(["cite","md"], _4);
  main.variable(observer()).define(["html","FileAttachment"], _5);
  main.variable(observer()).define(["cite","md"], _6);
  main.variable(observer()).define(["cite","md"], _7);
  main.variable(observer()).define(["html","FileAttachment"], _8);
  main.variable(observer()).define(["cite","md"], _9);
  main.variable(observer()).define(["md"], _10);
  main.variable(observer()).define(["html","FileAttachment"], _11);
  main.variable(observer()).define(["cite","md"], _12);
  main.variable(observer()).define(["md"], _13);
  main.variable(observer()).define(["md"], _14);
  main.variable(observer()).define(["html","FileAttachment"], _15);
  main.variable(observer()).define(["cite","md"], _16);
  main.variable(observer()).define(["cite","md"], _17);
  main.variable(observer()).define(["md"], _18);
  main.variable(observer()).define(["md"], _19);
  main.variable(observer()).define(["html","FileAttachment"], _20);
  main.variable(observer()).define(["md"], _21);
  main.variable(observer()).define(["cite","md"], _22);
  main.variable(observer()).define(["cite","tex","md"], _23);
  main.variable(observer()).define(["tex"], _24);
  main.variable(observer()).define(["tex"], _25);
  main.variable(observer()).define(["tex","md"], _26);
  main.variable(observer()).define(["tex"], _27);
  main.variable(observer()).define(["tex"], _28);
  main.variable(observer()).define(["tex"], _29);
  main.variable(observer()).define(["tex","md"], _30);
  main.variable(observer()).define(["tex","md"], _31);
  main.variable(observer()).define(["md"], _32);
  main.variable(observer()).define(["tex","md"], _33);
  main.variable(observer()).define(["tex","md"], _34);
  main.variable(observer()).define(["tex"], _35);
  main.variable(observer()).define(["tex","md"], _36);
  main.variable(observer()).define(["md"], _37);
  main.variable(observer()).define(["DOM","Plotly","data_to_plot"], _38);
  main.variable(observer("viewof range")).define("viewof range", ["Inputs"], _range);
  main.variable(observer("range")).define("range", ["Generators", "viewof range"], (G, _) => G.input(_));
  main.variable(observer()).define(["DOM","Plotly","data_to_plot_1"], _40);
  main.variable(observer()).define(["md"], _41);
  main.variable(observer()).define(["tex","md"], _42);
  main.variable(observer()).define(["tex"], _43);
  main.variable(observer()).define(["tex","md"], _44);
  main.variable(observer()).define(["md"], _45);
  main.variable(observer()).define(["tex"], _46);
  main.variable(observer()).define(["tex"], _47);
  main.variable(observer()).define(["tex","md"], _48);
  main.variable(observer()).define(["tex"], _49);
  main.variable(observer()).define(["tex","md"], _50);
  main.variable(observer()).define(["DOM","Plotly","pinn_to_plot_u"], _51);
  main.variable(observer()).define(["DOM","Plotly","pinn_to_plot_N"], _52);
  main.variable(observer()).define(["md"], _53);
  main.variable(observer()).define(["cite","md"], _54);
  main.variable(observer("viewof opt")).define("viewof opt", ["Inputs"], _opt);
  main.variable(observer("opt")).define("opt", ["Generators", "viewof opt"], (G, _) => G.input(_));
  main.variable(observer("viewof lr")).define("viewof lr", ["Inputs"], _lr);
  main.variable(observer("lr")).define("lr", ["Generators", "viewof lr"], (G, _) => G.input(_));
  main.variable(observer()).define(["DOM","Plotly","conv_plot_dd"], _57);
  main.variable(observer()).define(["DOM","Plotly","conv_plot_pinn"], _58);
  main.variable(observer()).define(["md"], _59);
  main.variable(observer()).define(["cite","md"], _60);
  main.variable(observer()).define(["width","DOM","Plotly","plot_loss_dd"], _61);
  main.variable(observer()).define(["unpack","z_data_pinn","width","DOM","Plotly"], _62);
  main.variable(observer()).define(["md"], _63);
  main.variable(observer()).define(["tex","md"], _64);
  main.variable(observer()).define(["tex"], _65);
  main.variable(observer("viewof loss_term")).define("viewof loss_term", ["Inputs"], _loss_term);
  main.variable(observer("loss_term")).define("loss_term", ["Generators", "viewof loss_term"], (G, _) => G.input(_));
  main.variable(observer("viewof data_points")).define("viewof data_points", ["Inputs"], _data_points);
  main.variable(observer("data_points")).define("data_points", ["Generators", "viewof data_points"], (G, _) => G.input(_));
  main.variable(observer()).define(["unpack","ground_truth_surface_n","get_z_data","loss_term","data_points","width","DOM","Plotly"], _68);
  main.variable(observer()).define(["tex","md"], _69);
  main.variable(observer()).define(["md"], _70);
  main.variable(observer()).define(["md"], _71);
  main.variable(observer()).define(["iterate_columns","ground_truth_surface_n","get_reliablity_data","radios","unpack","reliability_data","width","DOM","Plotly"], _72);
  main.variable(observer("viewof radios")).define("viewof radios", ["Inputs"], _radios);
  main.variable(observer("radios")).define("radios", ["Generators", "viewof radios"], (G, _) => G.input(_));
  main.variable(observer()).define(["unpack","ground_truth_surface_n","get_reliablity_pinn","radios","reliability_data","width","DOM","Plotly"], _74);
  main.variable(observer()).define(["md"], _75);
  main.variable(observer()).define(["md"], _76);
  main.variable(observer()).define(["md"], _77);
  main.variable(observer()).define(["md"], _78);
  main.variable(observer()).define(["bibliography"], _79);
  main.variable(observer()).define(["md"], _80);
  main.variable(observer("Plotly")).define("Plotly", ["require"], _Plotly);
  main.variable(observer("csv")).define("csv", ["Plotly"], _csv);
  main.variable(observer("unpack")).define("unpack", _unpack);
  main.variable(observer("get_loss_string")).define("get_loss_string", _get_loss_string);
  main.variable(observer("get_substring")).define("get_substring", _get_substring);
  main.variable(observer("is_legend")).define("is_legend", _is_legend);
  main.variable(observer("get_z_data")).define("get_z_data", ["get_loss_string","unpack","D20","D40","D60","D80","DB100","B100","P80","DB20","DB40","DB60","DB80","DP20","DP40","DP60","DP80","DP100","BP100","DBP20","DBP40","DBP60","DBP80","DBP100"], _get_z_data);
  main.variable(observer("get_reliablity_data")).define("get_reliablity_data", ["iterate_columns","reliability_dd_orig","reliability_dd_noise","reliability_dd_outlier","reliability_dd_ood"], _get_reliablity_data);
  main.variable(observer("get_reliablity_pinn")).define("get_reliablity_pinn", ["iterate_columns","reliability_pinn","reliability_pinn_noise","reliability_pinn_outlier","reliability_pinn_ood"], _get_reliablity_pinn);
  main.variable(observer("iterate_columns")).define("iterate_columns", ["unpack"], _iterate_columns);
  main.variable(observer("iterate_rows")).define("iterate_rows", _iterate_rows);
  const child1 = runtime.module(define1);
  main.import("bib", child1);
  main.variable(observer("viewof bibliography")).define("viewof bibliography", ["bib"], _bibliography);
  main.variable(observer("bibliography")).define("bibliography", ["Generators", "viewof bibliography"], (G, _) => G.input(_));
  main.variable(observer("cite")).define("cite", ["viewof bibliography"], _cite);
  main.variable(observer("data_to_plot")).define("data_to_plot", ["ground_truth_single_eq","range","unpack","dd_single_eq_obs"], _data_to_plot);
  main.variable(observer("data_to_plot_1")).define("data_to_plot_1", ["ground_truth_single_eq","range","unpack","dd_single_eq_obs"], _data_to_plot_1);
  main.variable(observer("pinn_to_plot_u")).define("pinn_to_plot_u", ["ground_truth_single_eq","unpack","pinn_single_eq"], _pinn_to_plot_u);
  main.variable(observer("pinn_to_plot_N")).define("pinn_to_plot_N", ["ground_truth_single_eq","unpack","pinn_single_eq"], _pinn_to_plot_N);
  main.variable(observer("conv_plot_dd")).define("conv_plot_dd", ["unpack","data_loss_adam_rms","opt","lr","data_lbfgs_end","get_substring","is_legend"], _conv_plot_dd);
  main.variable(observer("conv_plot_pinn")).define("conv_plot_pinn", ["unpack","pinn_loss_adam_rms","opt","lr","pinn_lbfgs_end","get_substring","is_legend"], _conv_plot_pinn);
  main.variable(observer("plot_loss_dd")).define("plot_loss_dd", ["iterate_columns","z_data_data"], _plot_loss_dd);
  const child2 = runtime.module(define2);
  main.import("py", child2);
  main.import("pyodide", child2);
  main.variable(observer("numpy")).define("numpy", ["pyodide","py"], _numpy);
  const child3 = runtime.module(define3);
  main.import("numpy", "np", child3);
  main.variable(observer("ground_truth_surface_u")).define("ground_truth_surface_u", ["__query","FileAttachment","invalidation"], _ground_truth_surface_u);
  main.variable(observer("ground_truth_surface_n")).define("ground_truth_surface_n", ["__query","FileAttachment","invalidation"], _ground_truth_surface_n);
  main.variable(observer("pinn_lbfgs_end")).define("pinn_lbfgs_end", ["__query","FileAttachment","invalidation"], _pinn_lbfgs_end);
  main.variable(observer("pinn_loss_adam_rms")).define("pinn_loss_adam_rms", ["__query","FileAttachment","invalidation"], _pinn_loss_adam_rms);
  main.variable(observer("ground_truth_single_eq")).define("ground_truth_single_eq", ["__query","FileAttachment","invalidation"], _ground_truth_single_eq);
  main.variable(observer("dd_single_eq_obs")).define("dd_single_eq_obs", ["__query","FileAttachment","invalidation"], _dd_single_eq_obs);
  main.variable(observer("pinn_single_eq")).define("pinn_single_eq", ["__query","FileAttachment","invalidation"], _pinn_single_eq);
  main.variable(observer("data_loss_adam_rms")).define("data_loss_adam_rms", ["__query","FileAttachment","invalidation"], _data_loss_adam_rms);
  main.variable(observer("data_lbfgs_end")).define("data_lbfgs_end", ["__query","FileAttachment","invalidation"], _data_lbfgs_end);
  main.variable(observer("z_data_data")).define("z_data_data", ["__query","FileAttachment","invalidation"], _z_data_data);
  main.variable(observer("z_data_pinn")).define("z_data_pinn", ["__query","FileAttachment","invalidation"], _z_data_pinn);
  main.variable(observer("D20")).define("D20", ["__query","FileAttachment","invalidation"], _D20);
  main.variable(observer("DB20")).define("DB20", ["__query","FileAttachment","invalidation"], _DB20);
  main.variable(observer("DP20")).define("DP20", ["__query","FileAttachment","invalidation"], _DP20);
  main.variable(observer("DBP20")).define("DBP20", ["__query","FileAttachment","invalidation"], _DBP20);
  main.variable(observer("D40")).define("D40", ["__query","FileAttachment","invalidation"], _D40);
  main.variable(observer("DB40")).define("DB40", ["__query","FileAttachment","invalidation"], _DB40);
  main.variable(observer("DP40")).define("DP40", ["__query","FileAttachment","invalidation"], _DP40);
  main.variable(observer("DBP40")).define("DBP40", ["__query","FileAttachment","invalidation"], _DBP40);
  main.variable(observer("D60")).define("D60", ["__query","FileAttachment","invalidation"], _D60);
  main.variable(observer("DB60")).define("DB60", ["__query","FileAttachment","invalidation"], _DB60);
  main.variable(observer("DP60")).define("DP60", ["__query","FileAttachment","invalidation"], _DP60);
  main.variable(observer("DBP60")).define("DBP60", ["__query","FileAttachment","invalidation"], _DBP60);
  main.variable(observer("D80")).define("D80", ["__query","FileAttachment","invalidation"], _D80);
  main.variable(observer("P80")).define("P80", ["__query","FileAttachment","invalidation"], _P80);
  main.variable(observer("DB80")).define("DB80", ["__query","FileAttachment","invalidation"], _DB80);
  main.variable(observer("DP80")).define("DP80", ["__query","FileAttachment","invalidation"], _DP80);
  main.variable(observer("DBP80")).define("DBP80", ["__query","FileAttachment","invalidation"], _DBP80);
  main.variable(observer("D100")).define("D100", ["__query","FileAttachment","invalidation"], _D100);
  main.variable(observer("B100")).define("B100", ["__query","FileAttachment","invalidation"], _B100);
  main.variable(observer("DB100")).define("DB100", ["__query","FileAttachment","invalidation"], _DB100);
  main.variable(observer("BP100")).define("BP100", ["__query","FileAttachment","invalidation"], _BP100);
  main.variable(observer("DP100")).define("DP100", ["__query","FileAttachment","invalidation"], _DP100);
  main.variable(observer("DBP100")).define("DBP100", ["__query","FileAttachment","invalidation"], _DBP100);
  main.variable(observer("reliability_data")).define("reliability_data", ["__query","FileAttachment","invalidation"], _reliability_data);
  main.variable(observer("reliability_dd_orig")).define("reliability_dd_orig", ["__query","FileAttachment","invalidation"], _reliability_dd_orig);
  main.variable(observer("reliability_dd_noise")).define("reliability_dd_noise", ["__query","FileAttachment","invalidation"], _reliability_dd_noise);
  main.variable(observer("reliability_dd_outlier")).define("reliability_dd_outlier", ["__query","FileAttachment","invalidation"], _reliability_dd_outlier);
  main.variable(observer("reliability_dd_ood")).define("reliability_dd_ood", ["__query","FileAttachment","invalidation"], _reliability_dd_ood);
  main.variable(observer("reliability_pinn")).define("reliability_pinn", ["__query","FileAttachment","invalidation"], _reliability_pinn);
  main.variable(observer("reliability_pinn_noise")).define("reliability_pinn_noise", ["__query","FileAttachment","invalidation"], _reliability_pinn_noise);
  main.variable(observer("reliability_pinn_outlier")).define("reliability_pinn_outlier", ["__query","FileAttachment","invalidation"], _reliability_pinn_outlier);
  main.variable(observer("reliability_pinn_ood")).define("reliability_pinn_ood", ["__query","FileAttachment","invalidation"], _reliability_pinn_ood);
  return main;
}
