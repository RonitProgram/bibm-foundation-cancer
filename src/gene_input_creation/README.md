# Gene input creation

hvg1000.py and hvg10000.py rank genes on the 80% training split and export all
sample rows with log2(x+1). create_litgenes1.py uses the recovered gProfiler mapping;
create_litgenes2.py uses membership recovered from the saved second-set header.
All accept --input and --output; HVG scripts also accept --labels.
Defaults read data/expression and write data/generated_inputs.
Historical reference notebooks are under provenance/reference_notebooks.
