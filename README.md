# COMET Tracking : Machine Learning Approaches

This is my personal repository of my work on COMET tracking using various 
machine learning approaches.  This work was carried out with Alex Rogozhnikov at 
Yandex Data Factory.  The simulation data was produced using the ICEDUST 
software framework. I orchestrated the simulations themselves across various 
batch computing farms, namely the ones at CC-IN2P3, Imperial College London, 
IHEP, and Tianhe-2.

## The COMET Experiment

The Coherent Muon to Electron (COMET) a next-generation particle physics 
experiment is designed to investigate charged lepton flavour violation (CLFV) by 
searching for muon to electron conversion on an aluminium nucleus.  This process 
is not allowed in the Standard Model of particle physics, but has very good 
sensitivity to Beyond the Standard Model physics. 

<p align="center">
    <img src="https://github.com/ewengillies/track-finding-yandex/blob/update_readme/images/phase_i_no_background.png" width="500">
</p>

### The Search for New Physics

COMET will take place in two phases. The first phase is designed to probe muon 
to electron conversion 100 times better than the current limit. This target 
limit will look for  a single event in 10<sup>15</sup> events.  To give some 
scale to this search, we could reach a similar sensitivity if we looked at one 
event per minute since the beginning of the universe (13.8 billion years ago).

Unfortunately, we do not have 13.8 billion years for our search.  To combat 
this, the COMET experiment is designed to probe millions of events per second 
for our elusive signal of new physics.  This leads to a *high intenisty* 
environment, i.e. one with many many particles flying around in the detector.

### The COMET Phase-I Beamline

COMET is designed to transform a high intensity proton beam (read: many protons 
per second) into the ideal environment to watch for our signal process.  To do 
so, it employs a clever collection of magnets, targets, and filters to create 
a high intensity muon beam.  These components form the "beamline" of the 
experiment.

<p align="center">
    <img src="https://github.com/ewengillies/track-finding-yandex/blob/update_readme/images/PhaseI_schematic_no_back.png" width="500">
</p>

The red arrow in the figure above represents the direction of the incoming 
proton beam.  These protons hit a fixed target and interact with the target 
material to produce pions.  This target is surrounded by a large magnet, whose 
magnetic field is designed to direct the pions towards the curved part of the 
diagram above.

These pions are unstable particles, and naturally decay into muons while they 
are flying through the curved section.  The curved section is also a collection 
of magnets, which are designed to filter away and undesirable particles.  This 
delivers a muon beam to the detector region, at the bottom left end of the 
diagram above.  The bottom right shows two of the detector systems in use for 
Phase-I.

### The Cylindrical Detector

Inside the detector region, the muon beam is collides with a number of thin, 
fixed aluminum disks. Some of the muons lose enough momentum to stop inside the 
target. Once they do, they form muonic atoms, where a muon replaces an electron 
inside an aluminium atom in the target. **This is the condition needed from 
which the signal process can occur.** This condition happens millions of times 
per second, and for the vast majority of the time, we expect non-signal things 
to happen.  Below, we can see the muons in green colliding the with silver 
stopping target disks.  The disks are surrounded by the detector.

<p align="center">
    <img src="https://github.com/ewengillies/track-finding-yandex/blob/update_readme/images/particles_on_aluminum.png" width="500">
</p>

The signal process would yield an electron with much higher energy than the
background processes. To spot the signal, we need to spot this electron. Both
the detector and the stopping target are surrounded by a large magnet, 
which causes particle trajectories to curve and fly in a helix shape.  The 
radius of the curve of the helix is larger for a particle with a high momentum
(or energy).

With enough momentum, a particle enters the detector, which is the cylindrical 
volume.  This cylindrical volume contains an array of coaxial wires. As the 
particles pass these wires, they deposit a small amount of electric charge on 
the wire, which is then readout and saved to a file.  These are referred to as 
*hits* on the wire.

## Track Finding as a Classification Problem

The track finding algorithm must group signal hits in a given event so that the 
track fitting algorithm can fit a trajectory to the hits. As a first step, the 
proposed algorithm filters out all background hits. The image below shows 
a signal track, blue, leaving the aluminium target in the middle, entering the 
cylindrical detector, and leaving hits on the wires it passes. All of the hits 
in red are from background particles.  

<p align="center">
    <img src="https://github.com/ewengillies/track-finding-yandex/blob/update_readme/images/paricle_through_detector.png" width="500">
</p>

### Features of a Hit

A hit is characterized by three main "local features" in this geometry: 
  * The amount of energy deposited, 
  * The time which the hit occurred,
  * The radial distance of the hit from the target.

By design, the amount of energy deposited is already a great feature for 
classification. Many of the background particles leaving the red hits are 
protons, which deposit more energy than electrons. Traditionally, physicists 
would cut on this feature as the basis of a classification algorithm. The 
picture below compares the normalized distributions of energy depositions for 
signal and background particles.  Note the logarithmic x-axis.  **The 
performance of the algorithm will often be compared to the performance of only 
using this feature, and not considering any others.**

<p align="center">
    <img src="https://github.com/ewengillies/track-finding-yandex/blob/update_readme/images/edep.png" width="500">
</p>

Classifying using this set of local features provides signifiant gains over 
using only the energy deposition.  To harvest even more classification power 
from these features, the properties of neighbouring hits are also considered.  
For each hit, this add four more features: 
 * The energy deposition on the wires to the left and right, if any
 * The timing of the hit on the wires to the left and right, if any.
These four features, combined with the original three local features, define 
a total set of 7 features, which I will refer to as the "neighbour feature" set.

### Classifying an Event

Lets illustrate the rest of the algorithm by way of an example.  First, lets 
start with our unfiltered event:

<p align="center">
    <img src="https://github.com/ewengillies/track-finding-yandex/blob/update_readme/images/unlabeled_event.png" width="500"/> <img src="https://github.com/ewengillies/track-finding-yandex/blob/update_readme/images/labelled_event.png" width="500"/>
</p>

<!---
Before these events are written to disk, the experiment is designed to filter 
out interactions that are clearly uninteresting, while saving the ones that look 
as if they may contain a signal electron. This mechanism is referred to as the 
trigger of the experiment.  Currently, the COMET trigger system is not refined 
enough, causing it to save a hundred times more events than the current data 
readout hardware can handle.  
-->
