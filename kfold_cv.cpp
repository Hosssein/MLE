
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
using namespace std;

double compute_avg(vector<double> vec){
	double size = vec.size() , sum = 0;
	for (int i = 0; i<vec.size() ; i++)
		sum += vec[i];
	return sum/size;
}
void k_fold_cross_validation(int k, int q_num, string filePath){
	ifstream in(filePath.c_str());
	string temp;
	int fold = q_num/k;
	//double thr;
	string thr;

	string oo = filePath+"_fmeasure_cv";

	ofstream out(oo.c_str());

	map<string, map<int,double> >avg_fold_prec,avg_fold_recall,avg_fold_f_measure;
	map<string, map<int,vector <double> > >thr_prec,thr_recall,thr_f_measure;
	map<int,double>fold_size;
	while(getline(in,thr)){
		int counter = 0;
		map<int,vector <double> >fold_prec,fold_recall,fold_f_measure;
		//getline(in,temp);
		//cout<<"thr: "<<thr<<endl;
		for(int i = 0 ; i<k ; i++){
			//cout<<i<<endl;
			for(int j = 0 ; j< fold; j++){
				getline(in,temp);
				stringstream ss(temp);
				double prec,recall;
				ss>>temp>>temp>>prec>>temp>>temp>>recall;
				//cout<<temp<<endl;
				fold_prec[i].push_back(prec);
				fold_recall[i].push_back(recall);
				counter++;
				if (prec == 0 && recall== 0 )
					fold_f_measure[i].push_back(0);	
				else
					fold_f_measure[i].push_back((2*prec*recall)/(prec+recall));
				//cout<<j<<" "<<prec<<" "<<recall<<endl;
			}
			fold_size[i] = fold_prec[i].size();
		}
		//edit for other than 2 fold!
		for (int i = fold*k ; i<q_num ; i++){
			getline(in,temp);
			stringstream ss(temp);
			double prec,recall;
			ss>>temp>>temp>>prec>>temp>>temp>>recall;
			//cout<<temp<<endl;
			counter++;
			fold_prec[k-1].push_back(prec);
			fold_recall[k-1].push_back(recall);
			if (prec == 0 && recall== 0 )
				fold_f_measure[k-1].push_back(0);	
			else
				fold_f_measure[k-1].push_back((2*prec*recall)/(prec+recall));
		}
		//cout<<"last "<<prec<<" "<<recall<<endl;
		//cout<<counter<<endl;
		string temp2,temp3;
		getline(in,temp3);
		getline(in,temp2);
		getline(in,temp);
		getline(in,temp);
		//
		getline(in,temp);
		getline(in,temp);
		getline(in,temp);
		getline(in,temp);
		getline(in,temp);
		//
		getline(in,temp);
		getline(in,temp);
		getline(in,temp);
		getline(in,temp);

		thr += temp3+temp2;
		fold_size[k-1] = fold_prec[k-1].size();
		for (int i = 0 ; i<k ; i++){
			avg_fold_prec[thr][i] = compute_avg(fold_prec[i]);
			avg_fold_recall[thr][i] = compute_avg(fold_recall[i]);
			avg_fold_f_measure[thr][i] = compute_avg(fold_f_measure[i]);
			//cout<<avg_fold_prec[thr][i]<<endl;
		}
		/*for (int i = 0 ; i<k ; i++){
			double sum_1 = 0,sum_2 = 0,sum_3 = 0;
			for (int j = 0 ; j<k ; j++){
				if (j!=i){
					//cout<<j<<endl;
					sum_1+= avg_fold_prec[thr][j]*fold_prec[j].size();
					sum_2+= avg_fold_recall[thr][j]*fold_prec[j].size();
					sum_3+= avg_fold_f_measure[thr][j]*fold_prec[j].size();
					//num+= fold_prec[j].size();
					//cout<<sum_3/num<<endl;
				}
			}
			double num = q_num - fold_prec[i].size();
			avg_fold_prec[thr][i] = sum_1/num;
			//cout<<avg_fold_prec[thr][i]<<endl;
			avg_fold_recall[thr][i] = sum_2/num;
			avg_fold_f_measure[thr][i] = sum_3/num;
		}*/
		thr_prec[thr] = fold_prec;
		thr_recall[thr] = fold_recall;
		thr_f_measure[thr] = fold_f_measure;
	}
	map<int, double> max_prec, max_recall,max_f_measure;
	map<int, string> max_prec_thr, max_recall_thr, max_f_measure_thr;
	for(int i = 0 ; i<k ; i++){
		max_prec[i] = -1;
		max_recall[i] = -1;
		max_f_measure[i] = -1;
	}
	for(int i = 0 ; i<k ; i++){
		//cout<<"<"<<i<<">"<<endl;
		for(map<string, map<int,double> >::iterator it = avg_fold_f_measure.begin() ; it!=avg_fold_f_measure.end() ; it++){
			double sum = 0, num = 0;
			for(int j = 0 ; j<k ; j++){
				if (j==i)
					continue;
				sum += it->second[j]* fold_size[j];
				num += fold_size[j];				
			}
			//cout<<sum<<" "<<num<<endl;
			if (sum/num > max_f_measure[i]){
				max_f_measure[i] = sum/num;
				max_f_measure_thr[i] = it->first;
			}
		}
	}
	double sum = 0, num = 0;
	for(int i = 0 ; i<k ; i++){
		cout<<i<<" "<<max_f_measure_thr[i]<<" "<<max_f_measure[i]<<endl;
		for(int j = 0 ; j<fold_size[i];j++){
			out<<thr_f_measure[max_f_measure_thr[i]][i][j]<<endl;
			sum +=	thr_f_measure[max_f_measure_thr[i]][i][j];
			num++;
		}
	}
	out<<"fmeasure: "<<sum/num<<endl;
	/*for(int i = 0 ; i<k ; i++){
		//cout<<"<"<<i<<">"<<endl;
		for(map<string, map<int,double> >::iterator it = avg_fold_prec.begin() ; it!=avg_fold_prec.end() ; it++){
			double sum = 0, num = 0;
			for(int j = 0 ; j<k ; j++){
				if (j==i)
					continue;
				sum += it->second[j]* fold_size[j];
				num += fold_size[j];				
			}
			//cout<<sum<<" "<<num<<endl;
			if (sum/num > max_prec[i]){
				max_prec[i] = sum/num;
				max_prec_thr[i] = it->first;
			}
		}
	}
	double sum = 0, num = 0;
	for(int i = 0 ; i<k ; i++){
		//cout<<i<<" "<<max_prec_thr[i]<<" "<<max_prec[i]<<endl;
		for(int j = 0 ; j<fold_size[i];j++){
			out<<thr_prec[max_prec_thr[i]][i][j]<<endl;
			sum +=	thr_prec[max_prec_thr[i]][i][j];
			num++;
		}
	}
	out<<"prec: "<<sum/num<<endl;*/
	/*for(int i = 0 ; i<k ; i++){
		//cout<<"<"<<i<<">"<<endl;
		for(map<string, map<int,double> >::iterator it = avg_fold_recall.begin() ; it!=avg_fold_recall.end() ; it++){
			double sum = 0, num = 0;
			for(int j = 0 ; j<k ; j++){
				if (j==i)
					continue;
				sum += it->second[j]* fold_size[j];
				num += fold_size[j];				
			}
			//cout<<sum<<" "<<num<<endl;
			if (sum/num > max_recall[i]){
				max_recall[i] = sum/num;
				max_recall_thr[i] = it->first;
			}
		}
	}
	double sum = 0, num = 0;
	for(int i = 0 ; i<k ; i++){
		//cout<<i<<" "<<max_recall_thr[i]<<" "<<max_recall[i]<<endl;
		for(int j = 0 ; j<fold_size[i];j++){
			out<<thr_recall[max_recall_thr[i]][i][j]<<endl;
			sum +=	thr_recall[max_recall_thr[i]][i][j];
			num++;
		}
	}
	out<<"recall: "<<sum/num<<endl;*/
}


int main(int argc, char* argv[]){
	k_fold_cross_validation(5,47,argv[1]);
	//k_fold_cross_validation(2,47,"/home/mozhdeh/Desktop/IR/RS-Framework/RS/out_2/NegColl_2500_Nofb_NoUpdatingThr_profDocThr:_0.5:3.5(0.1)");
}
