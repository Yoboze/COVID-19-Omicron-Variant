#############################################################################################
#  How to run the code																		#
#############################################################################################

for ((i=0; i<5; i++)); do python3 Constant.py > output_${i}.txt; done

OR

for job in 1 2 3 4 5; do python3 constant.py > output_${job}.txt; done

#############################################################################################
#  How to generate the txt files															#
#############################################################################################


cat output_1.txt | grep -e 'days' | cut -f 2- -d ' ' > days.txt
cat output_1.txt | grep -e 'cases' | cut -f 2- -d ' ' > cases.txt
cat *.txt | grep -e 'time' | cut -f 2- -d ' ' > time.txt
cat *.txt | grep -e 'infectd' | cut -f 2- -d ' ' > infectd.txt
cat *.txt | grep -e 'beta:' | cut -f 2- -d ' ' > beta.txt
cat *.txt | grep -e 'kappa:' | cut -f 2- -d ' ' > kappa.txt
cat *.txt | grep -e 'Nf:' | cut -f 2- -d ' ' > Nf.txt
cat *.txt | grep -e 'R_sq:' | cut -f 2- -d ' ' > R_sq.txt
cat *.txt | grep -e 'MAPE:' | cut -f 2- -d ' ' > MAPE.txt
cat *.txt | grep -e 'EV:' | cut -f 2- -d ' ' > EV.txt
cat *.txt | grep -e 'RMSE:' | cut -f 2- -d ' ' > RMSE.txt
cat *.txt | grep -e 'mse_train_loss' | cut -f 2- -d ' ' > mse_train_loss.txt
cat *.txt | grep -e 'rmse_train_loss' | cut -f 2- -d ' ' > rmse_train_loss.txt
cat *.txt | grep -e 'mse_validation_loss' | cut -f 2- -d ' ' > mse_validation_loss.txt
cat *.txt | grep -e 'rmse_validation_loss' | cut -f 2- -d ' ' > rmse_validation_loss.txt