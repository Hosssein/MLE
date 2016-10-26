/*==========================================================================
 * Copyright (c) 2001 Carnegie Mellon University.  All Rights Reserved.
 *
 * Use of the Lemur Toolkit for Language Modeling and Information Retrieval
 * is subject to the terms of the software license set forth in the LICENSE
 * file included with this software, and also available at
 * http://www.lemurproject.org/license.html
 *
 *==========================================================================
 */


#ifndef _TFIDFRETMETHOD_CPP
#define _TFIDFRETMETHOD_CPP
#include "TFIDFRetMethod.h"
#include "Param.hpp"
#include <cmath>
#include "Index.hpp"
#include "IndexManager.hpp"
#include "DocUnigramCounter.hpp"
#include "RelDocUnigramCounter.hpp"

using namespace lemur::api;
using namespace lemur::retrieval;
bool MF;
static int qid_piv=1;
TFIDFQueryRep::TFIDFQueryRep(const TermQuery &qry, const Index &dbIndex, double *idfValue, TFIDFParameter::WeightParam &param): ArrayQueryRep (dbIndex.termCountUnique()+1, qry, dbIndex), ind(dbIndex), idf(idfValue), prm(param)
{
  startIteration();
  while (hasMore()) {
    QueryTerm *qt = nextTerm();
    setCount(qt->id(), queryTFWeight(qt->weight())*idf[qt->id()]);
    // cout << "term : "<< dbIndex.term(qt->id()) << " idf="<< idf[qt->id()] <<    " total "<< dbIndex.docCount() << " with term "<< dbIndex.docCount(qt->id()) << endl;
    delete qt;
  }
}
double TFIDFQueryRep::queryTFWeight(const double rawTF) const
{
  if (prm.tf == TFIDFParameter::RAWTF) {
    return (rawTF);
  } else if (prm.tf == TFIDFParameter::LOGTF) {
    return (log(rawTF+1));
  } else if (prm.tf == TFIDFParameter::BM25) {
    return (TFIDFRetMethod::BM25TF(rawTF,prm.bm25K1,0,
                                   1, 1));  // no length normalization for query 
  } else {  // default to raw TF
    cerr << "Warning: unknown TF method, raw TF assumed\n";
    return rawTF;
  }
}



double TFIDFDocRep::docTFWeight(const double rawTF) const
{
  double s=0.1;
  if (prm.tf == TFIDFParameter::RAWTF) {
    return (rawTF);
  } else if (prm.tf == TFIDFParameter::LOGTF) {  
	return ((1 + log(1 + log(rawTF)))/((1-s) + s * ((docLength)/ind.docLengthAvg())));
    //return (log(rawTF+1));
  } else if (prm.tf == TFIDFParameter::BM25) {
    
    return (TFIDFRetMethod::BM25TF(rawTF, prm.bm25K1, prm.bm25B,
                                   docLength, ind.docLengthAvg()));
  } else {  // default to raw TF
    cerr << "Warning: unknown TF method, raw TF assumed\n";
    return rawTF;
  }
}


TFIDFRetMethod::TFIDFRetMethod(const Index &dbIndex, ScoreAccumulator &accumulator) :TextQueryRetMethod(dbIndex, accumulator) 
{
  // set default parameter value
 	docTFParam.tf = TFIDFParameter::BM25;
  //docTFParam.tf = TFIDFParameter::LOGTF;
  //docTFParam.tf = TFIDFParameter::RAWTF;
  docTFParam.bm25K1 = TFIDFParameter::defaultDocK1;
  docTFParam.bm25B = TFIDFParameter::defaultDocB;
  
  //qryTFParam.tf = TFIDFParameter::RAWTF;
  qryTFParam.tf = TFIDFParameter::BM25;
  //qryTFParam.tf = TFIDFParameter::LOGTF;
  qryTFParam.bm25K1 = TFIDFParameter::defaultQryK1;
  qryTFParam.bm25B = TFIDFParameter::defaultQryB;
 
  fbParam.howManyTerms =50;//; TFIDFParameter::defaultHowManyTerms;
  fbParam.posCoeff = 0.5;//TFIDFParameter::defaultPosCoeff;

  // pre-compute IDF values
  idfV = new double[dbIndex.termCountUnique()+1];
  for (COUNT_T i=1; i<=dbIndex.termCountUnique(); i++) {
    idfV[i] = log((dbIndex.docCount()+1)/(0.5+dbIndex.docCount(i)));
  }
  scFunc = new ScoreFunction();
  MF=false;
}




void TFIDFRetMethod::updateTextQuery(TextQueryRep &qryRep, const DocIDSet &relDocs)
{
   if(!MF){
	RegularUpdateTextQuery(qryRep,relDocs);
	return;
   }
  //cout<<"javid"<<endl;
	COUNT_T numTerms=ind.termCountUnique();  
	float * centroidVector = new float[numTerms+1];
	lemur::langmod::DocUnigramCounter *dCounter = new lemur::langmod::DocUnigramCounter(relDocs, ind);
	int feedUniqCounts=0;
	int docCounts=0;
	map<int,int> tids;
	map<int,int> CTF;
	map<int,int> tids_reverse;
	map<int,int> dids;
	map<int,int> dids_reverse;
	dCounter->startIteration();
	while (dCounter->hasMore()) {
		int wd; //dmf FIXME
		double wdCt;
		dCounter->nextCount(wd, wdCt);
		feedUniqCounts++;
		tids[wd]=feedUniqCounts;
		tids_reverse[feedUniqCounts]=wd;
	}
	relDocs.startIteration();
	while(relDocs.hasMore()){
		int id;
		double pr;
		relDocs.nextIDInfo(id,pr);
		docCounts++;
		dids[docCounts]=id;
		dids_reverse[id]=docCounts;
		TermInfoList *tList = ind.termInfoList(id);
		TermInfo *info;
		tList->startIteration();
    TFIDFDocRep *dr;
    dr = dynamic_cast<TFIDFDocRep *>(computeDocRep(id));
		while(tList->hasMore()) {
			info = tList->nextEntry();
			CTF[info->termID()]+= dr->docTFWeight(info->count());
		}
    delete dr;
		delete tList;
	}
	dids[docCounts+1]=0;
	dids_reverse[0]=docCounts+1;

	//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	double *distQuery = new double[numTerms+1];
	double *distQueryEst = new double[numTerms+1];
	double **distFeedbackEst = new double*[docCounts+1];

	for(int i=0;i<docCounts+1;i++){
		distFeedbackEst[i] = new  double[feedUniqCounts+1];
	}
	for(int i=1;i<docCounts+1;i++){
		for(int j=1;j<feedUniqCounts+1;j++){
			distFeedbackEst[i][j]=0;
		}
	}

	qryRep.startIteration();
	double qNum=0;
	while (qryRep.hasMore()) {
		QueryTerm *qt = qryRep.nextTerm();
		qNum++;
	}
	for(int j=1;j<feedUniqCounts+1;j++){
		distFeedbackEst[0][j] = 0;
	}

	qryRep.startIteration();
	while (qryRep.hasMore()) {
		QueryTerm *qt = qryRep.nextTerm();
//		distFeedbackEst[0][tids[qt->id()]]= dr->docTFWeight(1 ,idfV[qt->id()])*idfV[qt->id()];//totalCount*225;
		double qLenAvg = 3.0;
	  double s=0.1;
		//distFeedbackEst[0][tids[qt->id()]] =((1 + log(1 + log(1)))/((1-s) + s * ((qNum)/qLenAvg)))*idfV[qt->id()];
		//distFeedbackEst[0][tids[qt->id()]] = ((1.0)/((1-0.1) + 0.1 * ((qNum)/qLenAvg)))*idfV[qt->id()];//(log(1 + log (1 + 1)))*idfV[qt->id()];
//		distFeedbackEst[0][tids[qt->id()]] = (TFIDFRetMethod::BM25TF(1,TFIDFParameter::defaultQryK1,0, 1, 1)*idfV[qt->id()]);//+CTF[qt->id()]/docCounts)/2*idfV[qt->id()];
		distFeedbackEst[0][tids[qt->id()]] = qt->weight();// (log(1.0+1.0))*idfV[qt->id()];
		delete qt;
	}

	for (int i=1;i<=numTerms;i++) {
		centroidVector[i]=0;
	}

	relDocs.startIteration();
	int c=1;
	while(relDocs.hasMore()){
		int id;
		double pr;
		relDocs.nextIDInfo(id,pr);
		TermInfoList *tList = ind.termInfoList(id);
		TermInfo *info;
		tList->startIteration();
		TFIDFDocRep *dr= dynamic_cast<TFIDFDocRep *>(computeDocRep(id));
		while(tList->hasMore()) {
			info = tList->nextEntry(); 
			//distFeedbackEst[c][tids[info->termID()]]= log(info->count())*idfV[info->termID()];//totalCount*225;
			distFeedbackEst[c][tids[info->termID()]]= dr->docTFWeight(info->count())*idfV[info->termID()];//totalCount*225;
//			distFeedbackEst[c][tids[info->termID()]]= (dr->docTFWeight(info->count())*idfV[info->termID()]);//DF[info->termID()]//(log(1 + info->count()) + 1)*idfV[info->termID()];//totalCount*225;
			//cerr<<DF[info->termID()]<<endl;
			centroidVector[info->termID()]=1;//*idfV[info->termID()];
		}
		c++;
		delete tList;
	}

	char f[50];
	sprintf(f,"./model/%d",qid_piv);
	string s = f;
	ofstream out(s.c_str());
	out<<docCounts+1<<" "<<feedUniqCounts<<" "<<((docCounts+1)*feedUniqCounts)<<endl;

	for(int j=1;j<feedUniqCounts+1;j++){
		out<<(docCounts+1)<<" "<<j<<" "<<distFeedbackEst[0][j]<<endl;
	}


	for(int i=1;i<docCounts+1;i++){
		for(int j=1;j<feedUniqCounts+1;j++){
			out<<i<<" "<<j<<" "<<distFeedbackEst[i][j]<<endl;
		}
	}

	out.close();
	sprintf(f,"./model/tid_%d",qid_piv);
	s=f;
	ofstream out_tid(s.c_str());
	for(map<int,int>::iterator it=tids.begin();it!=tids.end();it++){
		out_tid<<it->second<<" "<<it->first<<endl;	
	}
	out_tid.close();
	sprintf(f,"./model/did_%d",qid_piv);
	s=f;
	ofstream out_did(s.c_str());
	for(map<int,int>::iterator it=dids.begin();it!=dids.end();it++){
		out_did<<it->second<<" "<<it->first<<endl;	
	}
	out_did.close();

	// Matrix Factorization Update
	double weight_bound = 0;
	sprintf(f,"./model/%d_UV.mm",qid_piv);
	s = f;
	cout<<"(o_O) "<<qid_piv<<endl;
	std::ifstream inf(s.c_str());
	std::string line;
	int i=0;
	map<int,map<int,double> > counts;

	// Read Estimated Values
	while(std::getline(inf,line)){
		istringstream ss(line);
		istream_iterator<string> begin(ss), end;
		vector<string> vqterms(begin,end);
		counts[atoi(vqterms[0].c_str())][atoi(vqterms[1].c_str())]=atof(vqterms[2].c_str());

	}
	map<int,double> k;
	for(int j=1;j<feedUniqCounts+1;j++){
		k[tids_reverse[j]]= counts[0][tids_reverse[j]];
	}

	vector <double> temp (numTerms+1);
	for (i=1; i<=numTerms; i++) {
		distQuery[i]=k[i];
		temp[i] = k[i];
	}
	sort(temp.begin(), temp.end());
	weight_bound = temp[numTerms-fbParam.howManyTerms];
	//	weight_bound = 0;
	//lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);
	/*int cc = 0;
	  for (int i=1; i<=numTerms; i++) {
	  if (distQuery[i] > (weight_bound)) {
	  cc++;
	  lmCounter.incCount(i,distQuery[i]);//(((distQuery[i]-1)*docParam.JMLambda*collectLM->prob(tids_reverse[i])/(1.0-docParam.JMLambda))/sum));
	  }
	  }*/

	//===============================================================

	double lam=0.8;
	for (i=1; i<= numTerms; i++) {
		//if (centroidVector[i]>0 ){//&& distQuery[i]>weight_bound) {
		//if(distQuery[i]>weight_bound){
			centroidVector[i] =distQuery[i];//lam*distQuery[i]+(1-lam)*centroidVector[i]*idfV[i]/docCounts;
		//}
		//else{
		//	centroidVector[i] = 0;
		//}
	}

	IndexedRealVector centVector(0);
	for (i=1; i< numTerms; i++) {
		if (centroidVector[i]>0) {
			IndexedReal entry;
			entry.ind = i;
			entry.val = centroidVector[i];
			centVector.push_back(entry);
		}
	}
	centVector.Sort();
	IndexedRealVector::iterator j;
	//cerr << "%%%%%%%%%%%%%%%%" << endl;
	//vector <pair < int, double> > qVector;
	double qLength = 0;
	qryRep.startIteration();
	while (qryRep.hasMore()){
		QueryTerm* qt = qryRep.nextTerm();
		//if(centroidVector[qt->id()]==0){
		//(dynamic_cast<TFIDFQueryRep *> (&qryRep))->setCount(qt->id(), 0);
		//qVector.push_back(pair <int, double> (qt->id(), qt->weight()));
		qLength += pow(qt->weight()/*/idfV[qt->id()]*/, 2);
		//}
		delete qt;
	}
	qLength = sqrt(qLength);
	int termCount=0;
	double fbLength = 0;
	for (j= centVector.begin();j!=centVector.end();j++) {
		if (termCount++ >= fbParam.howManyTerms) {
			break;
		} else {
			fbLength += pow((*j).val/*/idfV[(*j).ind]*/, 2);
		}
	}
	termCount = 0;
	fbLength = sqrt(fbLength);
	cerr << "^^^^ " << qLength << " " << fbLength << endl;
  for (j= centVector.begin();j!=centVector.end();j++) {
		if (termCount++ >= fbParam.howManyTerms) {
			break;
		} else {
			double lambda = 0.2; // lambda*f+(1-lambda)*q
			double val = (*j).val*(qLength/fbLength)*lambda/(1.0-lambda);
		//	cerr << "Q: " << val << endl;
			(dynamic_cast<TFIDFQueryRep *>(&qryRep))->incCount((*j).ind,val);//*fbParam.posCoeff);
		}
	}

	qid_piv++;
	delete dCounter;
	delete[] distQuery;
	delete[] distQueryEst;
	delete[] centroidVector;
}


void TFIDFRetMethod::RegularUpdateTextQuery(TextQueryRep &qryRep, const DocIDSet &relDocs){
  COUNT_T totalTerm=ind.termCountUnique();  
  float * centroidVector = new float[totalTerm+1]; // one extra for OOV

  COUNT_T i;
  for (i=1;i<=totalTerm;i++) {
    centroidVector[i]=0;
  }

  int actualDocs=0;
  relDocs.startIteration();
  while (relDocs.hasMore()) {
    int docID;
    double relPr;
    relDocs.nextIDInfo(docID, relPr);
    actualDocs++;

    TermInfoList *tList = ind.termInfoList(docID);
    tList->startIteration();
    while (tList->hasMore()) {
      TermInfo *info = tList->nextEntry();
      TFIDFDocRep *dr;
      dr = dynamic_cast<TFIDFDocRep *>(computeDocRep(docID));
      centroidVector[info->termID()] += dr->docTFWeight(info->count());
      delete dr;
    }
    delete tList;
  }

  // adjust term weight to obtain true Rocchio weight
  for (i=1; i<= totalTerm; i++) {
    if (centroidVector[i] >0 ) {
      centroidVector[i] *= idfV[i]/actualDocs;
    }
  }
	qryRep.startIteration();
	while (qryRep.hasMore()){
		QueryTerm* qt = qryRep.nextTerm();
		//(dynamic_cast<TFIDFQueryRep *> (&qryRep))->setCount(qt->id(), 0);// FIXME fb.Coef=1
		delete qt;
	}

  IndexedRealVector centVector(0);
  
  for (i=1; i< totalTerm; i++) {
    if (centroidVector[i]>0) {
      IndexedReal entry;
      entry.ind = i;
      entry.val = centroidVector[i];
      centVector.push_back(entry);
    }
  }
  centVector.Sort();
  IndexedRealVector::iterator j;
  int termCount=0;
  for (j= centVector.begin();j!=centVector.end();j++) {
    if (termCount++ >= fbParam.howManyTerms) {
      break;
    } else {
      // add the term to the query vector
			double lambda = 0.0;
			fbParam.posCoeff = lambda/(1.0-lambda);
      (dynamic_cast<TFIDFQueryRep *>(&qryRep))->incCount((*j).ind, (*j).val*fbParam.posCoeff);
    }
  }
	qid_piv++;
  delete[] centroidVector;

}

#endif
