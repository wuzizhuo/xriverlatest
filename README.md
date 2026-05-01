Receptor receives substances transmitted by other virtual cells; 
Response generates a new cell atlas under drug action; 
Growth factor produces drugs that drive cellular changes according to images of the initial and target cell states; 
Growth procession generates the dynamic pathway evolution process based on images of the initial and target cell states.

1.Cell self-evolve
Setting drug release to cell groups,example:T cells ,B cells, Microglia, Astrocyte to adjust their expression via gsvaldm. And employed lrldm to generate the next ligand-receptor interaction of pre cell groups to close the target/
ligand-receptor interaction such as mcao to normal issue.
python -/home/wupf_260213/controlnet/Cellhaness/IDgenerate/Cellevolve/smileasign.py
python -/home/wupf_260213/controlnet/Cellhaness/IDgenerate/Cellevolve/cellselfevolve.py

2.Cell growth factor
To explore the smile of cell change
python /home/wupf_260213/controlnet/ALLmodels/model/growth factor/drugldm/ldrugldm_train.py
python /home/wupf_260213/controlnet/ALLmodels/model/growth factor/drugldm/ldrugldm_generate_smiles.py

3.Cell process
To explore the continues gsva change between cellA state to cellB state under a smile under sapo
python /home/wupf_260213/controlnet/ALLmodels/model/growth process/Sapo/drugsapo.py

4.Respose
To explore the cell state A change direction under smile with gsvaldm 
python /home/wupf_260213/controlnet/ALLmodels/model/Response/gsvaldm/gsvaldm.py

5.Cell interaction
To explore the result of cell groups interaction 
python /home/wupf_260213/controlnet/Cellhaness/IDgenerate/CellMix/train_vae_ldm_pairformer.py

<img width="2185" height="2497" alt="Vitual" src="https://github.com/user-attachments/assets/a87a5141-2d0e-4b5f-9682-29f71a4888f4" />

