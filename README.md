# COMET Tracking : Machine Learning Approaches

This is my personal repository of my work on COMET tracking using various 
machine learning approaches.  This work was carried out with Alex Rogozhnikov at 
Yandex Data Factory.

## The COMET Experiment

The Coherent Muon to Electron (COMET) a next-generation particle physics 
experiment is designed to investigate charged lepton flavour violation (CLFV) by 
searching for muon to electron conversion on an aluminium nucleus.  This process 
is not allowed in the Standard Model of particle physics, but has very good 
sensitivity to Beyond the Standard Model physics. 

![phase-i](https://github.com/ewengillies/track-finding-yandex/blob/update_readme/images/phase_i_no_background.png | width=300)

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

![phase-i-schem](https://github.com/ewengillies/track-finding-yandex/blob/update_readme/images/PhaseI_schematic_no_back.png = 250x)


Before these events are written to disk, the experiment is designed to filter 
out interactions that are clearly uninteresting, while saving the ones that look 
as if they may contain a signal electron. This mechanism is referred to as the 
trigger of the experiment.  Currently, the COMET trigger system is not refined 
enough, causing it to save a hundred times more events than the current data 
readout hardware can handle.  
