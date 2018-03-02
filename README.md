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
The classification power of these features come from the fact that signal-like 
hits are often flanked by other signal-like hits. These four features, combined 
with the original three local features, define a total set of 7 features, which 
I will refer to as the "neighbour feature" set.

### Classifying an Event

Lets illustrate the rest of the algorithm by way of an example.  First, lets 
start with our unfiltered, unlabelled event:

<p align="center">
    <img src="https://github.com/ewengillies/track-finding-yandex/blob/update_readme/images/unlabeled_event.png" width="500"/>
</p>

Needless to say, its not at all obvious to the naked eye if this event contains 
a signal track, and which of these points form a signal track shape.  Lets add 
some labels to make it more obvious, where as before, blue is signal, red is 
background:

<p align="center">
    <img src="https://github.com/ewengillies/track-finding-yandex/blob/update_readme/images/labelled_event.png" width="500"/>
</p>

The signal points are surrounded by background points.  The first stage of 
classification is now used to improve the situation.  To this end, a GBDT is 
trained over the 7 neighbour features.  Each hit is classified, where a score of 
1 corresponds to a signal-like hit.  The fill of each hit is then scaled to this 
score, such that outlines with no fill mean a background-like response, whereas 
full circles indicate a signal like response:

<p align="center">
    <img src="https://github.com/ewengillies/track-finding-yandex/blob/update_readme/images/neighbour_class_event.png" width="500"/>
</p>

We can see in this event, this vastly improves our ability to spot the signal 
hits.  Most of the background points are filtered out, while most of the signal 
like hits remain.  With that said, there are still collections of background 
hits that are well separated from the signal track pattern. 

To fix these isolated background points, a shape-feature is created for each hit 
using a circular hough transform.  Essentially, the signal track radius is fed 
into the algorithm.  The space of potential track centres is discretized.  Each 
hit uses the signal track radius to determine which of the potential track 
centres its track could have originated from.  In this way, each hit "votes" on 
its favourite track centres.  This vote is weighted by the hits response from 
the first stage of the algorithm, such that signal-like hits get a higher vote. 
Graphically, we can picture this voting process as below.  In this image, the 
orange points are the potential track centres.  Their fill is weighted by how 
many votes they get.  Each green circle corresponds to one hit, where the 
overlap of this green circle with a track centre awards that track centre with 
votes: 

<p align="center">
    <img src="https://github.com/ewengillies/track-finding-yandex/blob/update_readme/images/hough_transform_event.png" width="500"/>
</p>

We can see that in this event, there are two distinct circles in the signal like 
track (due to the stereometry of the detector, which is not explained here). 
The algorithm is able to detect both of these as likely track centres. The trick 
now is to invert the transformation to allow the most likely track centres to 
pick out the hit points that they correspond to.  The "best" track centre can be 
determined by exponentially reweighing the voting score for each track centre, 
then to invert the mapping.  Graphically, this looks like: 

<p align="center">
    <img src="https://github.com/ewengillies/track-finding-yandex/blob/update_readme/images/inverse_hough_transform_event.png" width="500"/>
</p>

With each track centre now voting on its favourite hit points, each hit point 
can be assigned a score that describes "how likely the hit point is to be 
a signal-track radius away from the most likely track centres."  This defines 
our 8th feature.  Combined with the 7 neighbour features, our new feature set is 
called the "track feature" set.  The algorithm now trains a new GBDT over these 
8 features.  As before, we can visualize the output of this classifier on the 
sample event:

<p align="center">
    <img src="https://github.com/ewengillies/track-finding-yandex/blob/update_readme/images/classified_event.png" width="500"/>
</p>


<!---
Before these events are written to disk, the experiment is designed to filter 
out interactions that are clearly uninteresting, while saving the ones that look 
as if they may contain a signal electron. This mechanism is referred to as the 
trigger of the experiment.  Currently, the COMET trigger system is not refined 
enough, causing it to save a hundred times more events than the current data 
readout hardware can handle.  
-->
