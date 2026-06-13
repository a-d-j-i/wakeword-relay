# buenas noches → turn off 
python train.py --phrase "buenas noches" \
--samples 3000 --steps 30000 --neg_class_weight 12  --downloads_dir /vms2/work_tmp/download --data_dir /vms2/work_tmp/buenas_noches \
--piper_model es_AR-daniela-high es_MX-ald-medium es_ES-carlfm-x_low 

exit
# chispa magica → turn on 
python train.py --phrase "chispa magica" \
--samples 3000 --steps 30000 --neg_class_weight 12   --downloads_dir /vms2/work_tmp/download --data_dir /vms2/work_tmp/chispa_magica \
--piper_model es_AR-daniela-high es_MX-ald-medium es_ES-carlfm-x_low

