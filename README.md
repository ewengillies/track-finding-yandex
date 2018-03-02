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

COMET will take place in two phases. The first phase is designed to probe this 
process 100 times better than the current limit. This target limit will look for 
a single event in 10<sup>15</sup> events.


To reach its target 
sensitivity, COMET is designed to generate a million events per second, some of 
which will yield muon interactions, fewer of which will be relevant to muon to 
electron conversion.  

Before these events are written to disk, the experiment is designed to filter 
out interactions that are clearly uninteresting, while saving the ones that look 
as if they may contain a signal electron. This mechanism is referred to as the 
trigger of the experiment.  Currently, the COMET trigger system is not refined 
enough, causing it to save a hundred times more events than the current data 
readout hardware can handle.  
