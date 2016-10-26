/*==========================================================================
 *
 *  Original source copyright (c) 2001, Carnegie Mellon University.
 *  See copyright.cmu for details.
 *  Modifications copyright (c) 2002, University of Massachusetts.
 *  See copyright.umass for details.
 *
 *==========================================================================
 */



#include "RetMethod.h"
#include "Param.hpp"
#include "common_headers.hpp"
#include <cmath>
#include "DocUnigramCounter.hpp"
#include "RelDocUnigramCounter.hpp"
#include "IndexManager.hpp"
#include "OneStepMarkovChain.hpp"
#include <vector>
#include <set>
#include <algorithm>
#include <FreqVector.hpp>

using namespace lemur::api;
using namespace lemur::retrieval;
using namespace lemur::utility;


extern int RSMethodHM; // 0--> LM , 1--> RecSys
extern int negGenModeHM;//0 --> coll , 1--> nonRel
extern int feedbackMode;
extern int updatingThresholdMode; // 0 ->no updating ,1->linear

//extern map<int, map<int,double> > FEEDBACKMAP;
static int qid=1;
static string RM;
Index* myIndex = NULL;
void lemur::retrieval::QueryModel::interpolateWith(const lemur::langmod::UnigramLM &qModel,
                                                   double origModCoeff,
                                                   int howManyWord,
                                                   double prSumThresh,
                                                   double prThresh) {
    if (!qm) {
        qm = new lemur::api::IndexedRealVector();
    } else {
        qm->clear();
    }

    qModel.startIteration();
    while (qModel.hasMore()) {
        IndexedReal entry;
        qModel.nextWordProb((TERMID_T &)entry.ind,entry.val);
        qm->push_back(entry);

    }
    qm->Sort();

    double countSum = totalCount();

    // discounting the original model
    startIteration();
    while (hasMore()) {
        QueryTerm *qt = nextTerm();
        incCount(qt->id(), qt->weight()*origModCoeff/countSum);
        delete qt;
    }

    // now adding the new model
    double prSum = 0;
    int wdCount = 0;
    IndexedRealVector::iterator it;
    it = qm->begin();
    while (it != qm->end() && prSum < prSumThresh &&
           wdCount < howManyWord && (*it).val >=prThresh) {
        incCount((*it).ind, (*it).val*(1-origModCoeff));
        prSum += (*it).val;
        it++;
        wdCount++;
    }

    //Sum w in Q qtf * log(qtcf/termcount);
    colQLikelihood = 0;
    colQueryLikelihood();
    colKLComputed = false;
}

void lemur::retrieval::QueryModel::load(istream &is)
{
    // clear existing counts
    startIteration();
    QueryTerm *qt;
    while (hasMore()) {
        qt = nextTerm();
        setCount(qt->id(),0);
    }
    colQLikelihood = 0;

    int count;
    is >> count;
    char wd[500];
    double pr;
    while (count-- >0) {
        is >> wd >> pr;
        TERMID_T id = ind.term(wd);
        if (id != 0) setCount(id, pr); // don't load OOV terms
    }
    colQueryLikelihood();
    colKLComputed = false;
}

void lemur::retrieval::QueryModel::save(ostream &os)
{
    int count = 0;
    startIteration();
    QueryTerm *qt;
    while (hasMore()) {
        qt = nextTerm();
        count++;
        delete qt;
    }
    os << " " << count << endl;
    startIteration();
    while (hasMore()) {
        qt = nextTerm();
        os << ind.term(qt->id()) << " "<< qt->weight() << endl;
        delete qt;
    }
}

void lemur::retrieval::QueryModel::clarity(ostream &os)
{
    int count = 0;
    double sum=0, ln_Pr=0;
    startIteration();
    QueryTerm *qt;
    while (hasMore()) {
        qt = nextTerm();
        count++;
        // query-clarity = SUM_w{P(w|Q)*log(P(w|Q)/P(w))}
        // P(w)=cf(w)/|C|
        double pw = ((double)ind.termCount(qt->id())/(double)ind.termCount());
        // P(w|Q) is a prob computed by any model, e.g. relevance models
        double pwq = qt->weight();
        sum += pwq;
        ln_Pr += (pwq)*log(pwq/pw);
        delete qt;
    }
    // clarity should be computed with log_2, so divide by log(2).
    os << "=" << count << " " << (ln_Pr/(sum ? sum : 1.0)/log(2.0)) << endl;
    startIteration();
    while (hasMore()) {
        qt = nextTerm();
        // print clarity for each query term
        // clarity should be computed with log_2, so divide by log(2).
        os << ind.term(qt->id()) << " "
           << (qt->weight()*log(qt->weight()/
                                ((double)ind.termCount(qt->id())/
                                 (double)ind.termCount())))/log(2.0) << endl;
        delete qt;
    }
}

double lemur::retrieval::QueryModel::clarity() const
{
    int count = 0;
    double sum=0, ln_Pr=0;
    startIteration();
    QueryTerm *qt;
    while (hasMore()) {
        qt = nextTerm();
        count++;
        // query-clarity = SUM_w{P(w|Q)*log(P(w|Q)/P(w))}
        // P(w)=cf(w)/|C|
        double pw = ((double)ind.termCount(qt->id())/(double)ind.termCount());
        // P(w|Q) is a prob computed by any model, e.g. relevance models
        double pwq = qt->weight();
        sum += pwq;
        ln_Pr += (pwq)*log(pwq/pw);
        delete qt;
    }
    // normalize by sum of probabilities in the input model
    ln_Pr = ln_Pr/(sum ? sum : 1.0);
    // clarity should be computed with log_2, so divide by log(2).
    return (ln_Pr/log(2.0));
}

lemur::retrieval::RetMethod::RetMethod(const Index &dbIndex,
                                       const string &supportFileName,
                                       ScoreAccumulator &accumulator) :
    TextQueryRetMethod(dbIndex, accumulator), supportFile(supportFileName) {

    //docParam.smthMethod = RetParameter::defaultSmoothMethod;
    docParam.smthMethod = RetParameter::DIRICHLETPRIOR;
    //docParam.smthMethod = RetParameter::ABSOLUTEDISCOUNT;
    //docParam.smthMethod = RetParameter::JELINEKMERCER;

    docParam.smthStrategy= RetParameter::defaultSmoothStrategy;
    //docParam.ADDelta = RetParameter::defaultADDelta;
    docParam.JMLambda = RetParameter::defaultJMLambda;
    //docParam.JMLambda = 0.9;
    docParam.DirPrior = dbIndex.docLengthAvg();//50;//RetParameter::defaultDirPrior;

    qryParam.adjScoreMethod = RetParameter::NEGATIVEKLD;
    //qryParam.adjScoreMethod = RetParameter::QUERYLIKELIHOOD;
    //qryParam.fbMethod = RetParameter::defaultFBMethod;
    //qryParam.fbMethod = RetParameter::DIVMIN;




    qryParam.fbMethod = RetParameter::MIXTURE;
    RM="KUN";// *** Query Likelihood adjusted score method *** //
    //qryParam.fbCoeff = RetParameter::defaultFBCoeff;
    qryParam.fbCoeff =0.5;
    qryParam.fbPrTh = RetParameter::defaultFBPrTh;
    qryParam.fbPrSumTh = RetParameter::defaultFBPrSumTh;
    qryParam.fbTermCount = 5;//RetParameter::defaultFBTermCount;
    qryParam.fbMixtureNoise = RetParameter::defaultFBMixNoise;
    qryParam.emIterations = 50;//RetParameter::defaultEMIterations;

    docProbMass = NULL;
    uniqueTermCount = NULL;
    mcNorm = NULL;

    NegMu = ind.docLengthAvg();
    collectLMCounter = new lemur::langmod::DocUnigramCounter(ind);
    collectLM = new lemur::langmod::MLUnigramLM(*collectLMCounter, ind.termLexiconID());

    delta = 0.007;

    newNonRelRecieved = false;
    newRelRecieved = false;
    newNonRelRecievedCnt = 0,newRelRecievedCnt =0;

    switch (RSMethodHM)
    {
    case 0://lm
        // setThreshold(-4.3);
        setThreshold(-6.3);

        break;
    case 1://negGen
    {
        if(negGenModeHM==0)//col//mu=2500
            // setThreshold(2.1);
            setThreshold(1.5);
        else if(negGenModeHM == 1)
            // setThreshold(2.4);
            setThreshold(1.6);
        break;
    }
    }

    //prev_distQuery = new double[ind.termCountUnique()+1];
    scFunc = new ScoreFunc();
    scFunc->setScoreMethod(qryParam.adjScoreMethod);
}

lemur::retrieval::RetMethod::~RetMethod()
{
    //delete [] prev_distQuery;
    delete [] docProbMass;
    delete [] uniqueTermCount;
    delete [] mcNorm;
    delete collectLM;
    delete collectLMCounter;
    delete scFunc;
}

void lemur::retrieval::RetMethod::loadSupportFile() {
    ifstream ifs;
    int i;

    // Only need to load this file if smooth strategy is back off
    // or the smooth method is absolute discount. Don't reload if
    // docProbMass is not NULL.

    if (docProbMass == NULL &&
            (docParam.smthMethod == RetParameter::ABSOLUTEDISCOUNT ||
             docParam.smthStrategy == RetParameter::BACKOFF)) {
        cerr << "lemur::retrieval::SimpleKLRetMethod::loadSupportFile loading "
             << supportFile << endl;

        ifs.open(supportFile.c_str());
        if (ifs.fail()) {
            throw  Exception("lemur::retrieval::SimpleKLRetMethod::loadSupportFile",
                             "smoothing support file open failure");
        }
        COUNT_T numDocs = ind.docCount();
        docProbMass = new double[numDocs+1];
        uniqueTermCount = new COUNT_T[numDocs+1];
        for (i = 1; i <= numDocs; i++) {
            DOCID_T id;
            int uniqCount;
            double prMass;
            ifs >> id >> uniqCount >> prMass;
            if (id != i) {
                throw  Exception("lemur::retrieval::SimpleKLRetMethod::loadSupportFile",
                                 "alignment error in smooth support file, wrong id:");
            }
            docProbMass[i] = prMass;
            uniqueTermCount[i] = uniqCount;
        }
        ifs.close();
    }

    // only need to load this file if the feedback method is
    // markov chain. Don't reload if called a second time.

    if (mcNorm == NULL && qryParam.fbMethod == RetParameter::MARKOVCHAIN) {
        string mcSuppFN = supportFile + ".mc";
        cerr << "lemur::retrieval::SimpleKLRetMethod::loadSupportFile loading " << mcSuppFN << endl;

        ifs.open(mcSuppFN.c_str());
        if (ifs.fail()) {
            throw Exception("lemur::retrieval::SimpleKLRetMethod::loadSupportFile",
                            "Markov chain support file can't be opened");
        }

        mcNorm = new double[ind.termCountUnique()+1];

        for (i = 1; i <= ind.termCountUnique(); i++) {
            TERMID_T id;
            double norm;
            ifs >> id >> norm;
            if (id != i) {
                throw Exception("lemur::retrieval::SimpleKLRetMethod::loadSupportFile",
                                "alignment error in Markov chain support file, wrong id:");
            }
            mcNorm[i] = norm;
        }
    }
}

DocumentRep *lemur::retrieval::RetMethod::computeDocRep(DOCID_T docID)
{
    switch (docParam.smthMethod) {
    case RetParameter::JELINEKMERCER:
        return( new JMDocModel(docID,
                               ind.docLength(docID),
                               *collectLM,
                               docProbMass,
                               docParam.JMLambda,
                               docParam.smthStrategy));
    case RetParameter::DIRICHLETPRIOR:
        return (new DPriorDocModel(docID,
                                   ind.docLength(docID),
                                   *collectLM,
                                   docProbMass,
                                   docParam.DirPrior,
                                   docParam.smthStrategy));
    case RetParameter::ABSOLUTEDISCOUNT:
        return (new ABSDiscountDocModel(docID,
                                        ind.docLength(docID),
                                        *collectLM,
                                        docProbMass,
                                        uniqueTermCount,
                                        docParam.ADDelta,
                                        docParam.smthStrategy));
    case RetParameter::TWOSTAGE:
        return (new TStageDocModel(docID,
                                   ind.docLength(docID),
                                   *collectLM,
                                   docProbMass,
                                   docParam.DirPrior, // 1st stage mu
                                   docParam.JMLambda, // 2nd stage lambda
                                   docParam.smthStrategy));


    default:
        // this should throw, not exit.
        cerr << "Unknown document language model smoothing method\n";
        exit(1);
    }
}


void lemur::retrieval::RetMethod::updateProfile(lemur::api::TextQueryRep &origRep,
                                                vector<int> relJudgDoc ,vector<int> nonRelJudgDoc)
{
    //cerr<<"hahahaha"<<endl;

    IndexedRealVector rel , nonRel;
    for (int i =0 ; i<relJudgDoc.size() ; i++)
    {
        rel.PushValue(relJudgDoc[i],0);
    }
    for (int i =0 ; i<nonRelJudgDoc.size() ; i++)
    {
        nonRel.PushValue(nonRelJudgDoc[i],0);
    }
    PseudoFBDocs *relDocs , *nonRelDocs;
    relDocs= new PseudoFBDocs(rel,-1,true);
    nonRelDocs= new PseudoFBDocs(nonRel,-1,true);
    updateTextQuery(origRep,*relDocs,*nonRelDocs);
    delete relDocs;
    delete nonRelDocs;
}
void lemur::retrieval::RetMethod::updateThreshold(lemur::api::TextQueryRep &origRep,
                                                  vector<int> relJudgDoc ,vector<int> nonReljudgDoc , int mode/*,double relSumScores ,double nonRelSumScore*/)
{
    thresholdUpdatingMethod = updatingThresholdMode;
    //double alpha = 0.3,beta = 0.9;
    if(thresholdUpdatingMethod == 0)//no updating
        return;
    else if(thresholdUpdatingMethod == 1)//linear
    {
        if(mode == 0)//non rel passed
        {
            setThreshold(getThreshold()+getC1());
            //cout<<"mode 0 "<<getThreshold()<<endl;
        }
        else if(mode == 1)//not showed anything
        {
            setThreshold( getThreshold()- getC2() );
            //cout<<"mode 1 "<<getThreshold()<<endl;
        }

        //threshold = -4.5;
    }

    /*else if (thresholdUpdatingMethod == 2)//diff rel nonrel method
    {
        double alpha = getDiffThrUpdatingParam();
        double relSize = relJudgDoc.size();
        double nonRelSize = nonReljudgDoc.size();
        double val =  alpha * std::max( ((relSumScores/(relSize+1)+0.005) - (nonRelSumScore/(nonRelSize+1)+0.005)) ,-3.5) *
                (std::abs(std::log10( (nonRelSize+1) / (relSize+1) ) + 0.005 ) );


        if(mode == 0)
            setThreshold(getThreshold() + val);
        else
            setThreshold(getThreshold() - val);
        //cout<<relSumScores<<" "<<relSize<<endl;
        //cout<<alpha<<" "<<(relSumScores/(relSize+1.0))<<" "<<(nonRelSumScore/nonRelSize+1)<<" "<<(std::log10( (nonRelSize+1) / (relSize+1) ) + 0.005 )<<endl;
        cout <<"mode "<<mode<<" alpha "<<alpha <<" relSum: "<<(relSumScores/(relSize+1)+0.005)<<" nonRelSum: "<< (nonRelSumScore/(nonRelSize+1)+0.005) <<" val: "<<val<<" log: "<<std::log10( (nonRelSize+1) / (relSize+1) );
        cout<<" thr: "<<getThreshold()<<endl;
    }*/


}

extern vector<double>hquery;
extern vector<double>hrel;
extern vector<double>hnonRel;
extern vector<double>diagonal;
extern map<string ,vector<string> > queryRelTrainedDocsMap;
double  lemur::retrieval::RetMethod::ltuDocWeighting(int termID , int docID  )
{

    lemur::utility::HashFreqVector hfv(ind,docID);
    int tf = 0;
    hfv.find(termID , tf);
    if(tf)
    {
        double L = 1.0 + log10(tf);
        double T = log10((double)(ind.docCount()+1.0) / ind.docCount(termID) ) ;

        double avgUniqueTermPerDoc = ind.termCountUnique()/ ind.docCount();
        double U = 1.0 / (0.8 +0.2*( (double)(hfv.size()) / avgUniqueTermPerDoc ) );

        return L*T*U;
    }
    return -1.0;

}

void lemur::retrieval::RetMethod::initQueryVec(string qid, TextQueryRep *textQR)
{

    /*********************TrainDoc*****************************/
    hquery.clear();

    const map<string, vector<string> >::iterator fit = queryRelTrainedDocsMap.find(qid);

    const int trainDoc = 2;
    if( fit != queryRelTrainedDocsMap.end() )
    {
        for(int i = 0 ; i < trainDoc ; i++)
        {
            int docid = ind.document(fit->second[i] );
            TermInfoList *tList = ind.termInfoList( docid ) ;

            tList->startIteration();
            while (tList->hasMore())
            {
                TermInfo *info = tList->nextEntry();
                double ltu = ltuDocWeighting( info->termID() ,docid );

                hquery[info->termID()] += ltu;


                delete info;
            }
            delete tList;
        }
    }
    for(int i = 1 ; i < ind.termCountUnique() ; i++)
        hquery[i] /= (double)trainDoc;


    /*********************Query*****************************/
    double avgUniqueTermPerDoc = ind.termCountUnique()/ (double)(ind.docCount());
    set<int> qset;
    textQR->startIteration();
    while(textQR->hasMore())
    {
        QueryTerm *qt = textQR->nextTerm();
        qset.insert(qt->id()) ;
        delete qt;
    }

    textQR->startIteration();
    while(textQR->hasMore())
    {
        QueryTerm *qt = textQR->nextTerm();

        double tf = qt->weight();
        double L = 1.0 + log10(tf);
        double T = log10((double)(ind.docCount()+1.0) / ind.docCount(qt->id() ) ) ;


        double U = 1.0 / (0.8 + 0.2*( (double)qset.size() / avgUniqueTermPerDoc ) );


        hquery[qt->id()] += (L*T*U);

        delete qt;

    }

    for(int i = 1 ; i < ind.termCountUnique() ; i++)
        diagonal[i] = log10( ind.docCount() + 1 / (double)(ind.docCount(i)) );

    for(int i = 1 ; i < ind.termCountUnique() ; i++)
        hquery[i] = hquery[i] * diagonal[i];

}
float lemur::retrieval::RetMethod::computeProfDocSim(lemur::api::TextQueryRep *textQR,int docID ,
                                                     vector<int> relJudgDoc ,vector<int> nonReljudgDoc , bool newNonRel , bool newRel)
{

    const int totalTermNumber = ind.termCountUnique()+1;
    vector<double>tempDoc( totalTermNumber, 0.0);

    TermInfoList *tList = ind.termInfoList( docID ) ;
    tList->startIteration();
    while (tList->hasMore())
    {
        TermInfo *info = tList->nextEntry();
        double ltu = ltuDocWeighting( info->termID() ,docID );
        tempDoc[info->termID()] += ltu;

        delete info;
    }
    delete tList;

    float score = 0.0;
    for(int i = 1 ; i <= totalTermNumber ; i++)
    {
        score += hquery[i]*diagonal[i]*tempDoc[i];
    }
    cerr<<score<<endl;

    return score;

}


void lemur::retrieval::RetMethod::updateTextQuery(TextQueryRep &origRep,
                                                  const DocIDSet &relDocs,const DocIDSet &nonRelDocs )
{
    //cerr<<"fffffffffff"<<endl;
    QueryModel *qr;

    qr = dynamic_cast<QueryModel *> (&origRep);

    if(RM == "KUN")
    {
        computeKUNFBModel(*qr, relDocs,nonRelDocs);
        return;
    }else if(RM=="RM1"){
        computeRM1FBModel(*qr, relDocs,nonRelDocs);
        return;
    }else if(RM=="RM2"){
        computeRM2FBModel(*qr, relDocs);
        return;
    }else if(RM=="RM3"){
        computeRM3FBModel(*qr, relDocs);
        return;
    }else if(RM=="RM4"){
        computeRM4FBModel(*qr, relDocs);
        return;
    }else if(RM=="MIX"){
        computeMixtureFBModel(*qr, relDocs,nonRelDocs);
        return;
    }else if(RM=="DIVMIN"){
        computeDivMinFBModel(*qr, relDocs);
        return;
    }else if(RM=="MEDMM"){
        computeMEDMMFBModel(*qr, relDocs);
        return;
    }





    switch (qryParam.fbMethod) {
    case RetParameter::MIXTURE:
        computeMixtureFBModel(*qr, relDocs,nonRelDocs);
        break;
    case RetParameter::DIVMIN:
        computeDivMinFBModel(*qr, relDocs);
        break;
    case RetParameter::MARKOVCHAIN:
        computeMarkovChainFBModel(*qr, relDocs);
        break;
    case RetParameter::RM1:
        computeRM1FBModel(*qr, relDocs,nonRelDocs);
        break;
    case RetParameter::RM2:
        computeRM2FBModel(*qr, relDocs);
        break;
    default:
        throw Exception("SimpleKLRetMethod", "unknown feedback method");
        break;
    }
}


void lemur::retrieval::RetMethod::computeMixtureFBModel(QueryModel &origRep,
                                                        const DocIDSet &relDocs, const DocIDSet &nonRelDocs )
{
    COUNT_T numTerms = ind.termCountUnique();

    lemur::langmod::DocUnigramCounter *dCounter = new lemur::langmod::DocUnigramCounter(relDocs, ind);

    double *distQuery = new double[numTerms+1];
    double *distQueryEst = new double[numTerms+1];

    double noisePr;

    int i;

    double meanLL=1e-40;
    double distQueryNorm=0;

    for (i=1; i<=numTerms;i++) {
        distQueryEst[i] = rand()+0.001;
        distQueryNorm+= distQueryEst[i];
    }
    noisePr = qryParam.fbMixtureNoise;

    int itNum = qryParam.emIterations;
    do {
        // re-estimate & compute likelihood
        double ll = 0;

        for (i=1; i<=numTerms;i++) {

            distQuery[i] = distQueryEst[i]/distQueryNorm;
            // cerr << "dist: "<< distQuery[i] << endl;
            distQueryEst[i] =0;
        }

        distQueryNorm = 0;

        // compute likelihood
        dCounter->startIteration();
        while (dCounter->hasMore()) {
            int wd; //dmf FIXME
            double wdCt;
            dCounter->nextCount(wd, wdCt);
            ll += wdCt * log (noisePr*collectLM->prob(wd)  // Pc(w)
                              + (1-noisePr)*distQuery[wd]); // Pq(w)
        }
        meanLL = 0.5*meanLL + 0.5*ll;
        if (fabs((meanLL-ll)/meanLL)< 0.0001) {
            cerr << "converged at "<< qryParam.emIterations - itNum+1
                 << " with likelihood= "<< ll << endl;
            break;
        }

        // update counts

        dCounter->startIteration();
        while (dCounter->hasMore()) {
            int wd; // dmf FIXME
            double wdCt;
            dCounter->nextCount(wd, wdCt);

            double prTopic = (1-noisePr)*distQuery[wd]/
                    ((1-noisePr)*distQuery[wd]+noisePr*collectLM->prob(wd));

            double incVal = wdCt*prTopic;
            distQueryEst[wd] += incVal;
            distQueryNorm += incVal;
        }
    } while (itNum-- > 0);

    lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);
    for (i=1; i<=numTerms; i++) {
        if (distQuery[i] > 0) {
            lmCounter.incCount(i, distQuery[i]);
        }
    }
    lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(lmCounter, ind.termLexiconID());
    origRep.interpolateWith(*fblm, (1-qryParam.fbCoeff), qryParam.fbTermCount,
                            qryParam.fbPrSumTh, qryParam.fbPrTh);
    delete fblm;
    delete dCounter;
    delete[] distQuery;
    delete[] distQueryEst;

}


void lemur::retrieval::RetMethod::computeDivMinFBModel(QueryModel &origRep,
                                                       const DocIDSet &relDocs)
{
    COUNT_T numTerms = ind.termCountUnique();

    double * ct = new double[numTerms+1];

    TERMID_T i;
    for (i=1; i<=numTerms; i++) ct[i]=0;

    COUNT_T actualDocCount=0;
    relDocs.startIteration();
    while (relDocs.hasMore()) {
        actualDocCount++;
        int id;
        double pr;
        relDocs.nextIDInfo(id,pr);
        DocModel *dm;
        dm = dynamic_cast<DocModel *> (computeDocRep(id));

        for (i=1; i<=numTerms; i++) { // pretend every word is unseen
            ct[i] += log(dm->unseenCoeff()*collectLM->prob(i));
        }

        TermInfoList *tList = ind.termInfoList(id);
        TermInfo *info;
        tList->startIteration();
        while (tList->hasMore()) {
            info = tList->nextEntry();
            ct[info->termID()] += log(dm->seenProb(info->count(), info->termID())/
                                      (dm->unseenCoeff()*collectLM->prob(info->termID())));
        }
        delete tList;
        delete dm;
    }
    if (actualDocCount==0) return;

    lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);

    double norm = 1.0/(double)actualDocCount;
    for (i=1; i<=numTerms; i++) {
        lmCounter.incCount(i,
                           exp((ct[i]*norm -
                                qryParam.fbMixtureNoise*log(collectLM->prob(i)))
                               / (1.0-qryParam.fbMixtureNoise)));
    }
    delete [] ct;


    lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(lmCounter, ind.termLexiconID());
    origRep.interpolateWith(*fblm, (1-qryParam.fbCoeff), qryParam.fbTermCount,
                            qryParam.fbPrSumTh, qryParam.fbPrTh);
    delete fblm;
}
void lemur::retrieval::RetMethod::computeMEDMMFBModel(QueryModel &origRep,
                                                      const DocIDSet &relDocs)
{
    // Write Your own MEDMM right here
}
void lemur::retrieval::RetMethod::computeMarkovChainFBModel(QueryModel &origRep, const DocIDSet &relDocs)
{
    int stopWordCutoff =50;

    lemur::utility::ArrayCounter<double> *counter = new lemur::utility::ArrayCounter<double>(ind.termCountUnique()+1);

    lemur::langmod::OneStepMarkovChain * mc = new lemur::langmod::OneStepMarkovChain(relDocs, ind, mcNorm,
                                                                                     1-qryParam.fbMixtureNoise);
    origRep.startIteration();
    double summ;
    while (origRep.hasMore()) {
        QueryTerm *qt;
        qt = origRep.nextTerm();
        summ =0;
        mc->startFromWordIteration(qt->id());
        // cout << " +++++++++ "<< ind.term(qt->id()) <<endl;
        TERMID_T fromWd;
        double fromWdPr;

        while (mc->hasMoreFromWord()) {
            mc->nextFromWordProb(fromWd, fromWdPr);
            if (fromWd <= stopWordCutoff) { // a stop word
                continue;
            }
            summ += qt->weight()*fromWdPr*collectLM->prob(fromWd);
            // summ += qt->weight()*fromWdPr;
        }
        if (summ==0) {
            // query term doesn't exist in the feedback documents, skip
            continue;
        }

        mc->startFromWordIteration(qt->id());
        while (mc->hasMoreFromWord()) {
            mc->nextFromWordProb(fromWd, fromWdPr);
            if (fromWd <= stopWordCutoff) { // a stop word
                continue;
            }

            counter->incCount(fromWd,
                              (qt->weight()*fromWdPr*collectLM->prob(fromWd)/summ));
            // counter->incCount(fromWd, (qt->weight()*fromWdPr/summ));

        }
        delete qt;
    }
    delete mc;

    lemur::langmod::UnigramLM *fbLM = new lemur::langmod::MLUnigramLM(*counter, ind.termLexiconID());

    origRep.interpolateWith(*fbLM, 1-qryParam.fbCoeff, qryParam.fbTermCount,
                            qryParam.fbPrSumTh, qryParam.fbPrTh);
    delete fbLM;
    delete counter;
}

void lemur::retrieval::RetMethod::computeRM1FBModel(QueryModel &origRep,
                                                    const DocIDSet &relDocs,const DocIDSet &nonRelDocs)
{
    COUNT_T numTerms = ind.termCountUnique();

    // RelDocUnigramCounter computes SUM(D){P(w|D)*P(D|Q)} for each w
    lemur::langmod::RelDocUnigramCounter *dCounter = new lemur::langmod::RelDocUnigramCounter(relDocs, ind);
    lemur::langmod::RelDocUnigramCounter *nCounter = new lemur::langmod::RelDocUnigramCounter(nonRelDocs, ind);

    double *distQuery = new double[numTerms+1];
    double *negDistQuery = new double[numTerms+1];
    double expWeight = qryParam.fbCoeff;

    //double negWeight = 0.5;

    TERMID_T i;
    for (i=1; i<=numTerms;i++){
        distQuery[i] = 0.0;
        negDistQuery[i] = 0.0;
    }

    double pSum=0.0;
    dCounter->startIteration();
    while (dCounter->hasMore()) {
        int wd; // dmf FIXME
        double wdPr;
        dCounter->nextCount(wd, wdPr);
        distQuery[wd]=wdPr;
        pSum += wdPr;
    }
    double nSum=0.0;
    nCounter->startIteration();
    while (nCounter->hasMore()) {
        int wd; // dmf FIXME
        double wdPr;
        nCounter->nextCount(wd, wdPr);
        negDistQuery[wd]=wdPr;
        nSum += wdPr;
    }


    for (i=1; i<=numTerms;i++) {
        //REMOVE  2 *
        if(feedbackMode == 2)
        {
            cout<<"normalFB"<<endl;
            distQuery[i] = expWeight*distQuery[i]/pSum +
                    (1-expWeight)*ind.termCount(i)/ind.termCount();

        }else if(feedbackMode == 1)
        {
            cout<<"ourFB"<<endl;
            distQuery[i] =  expWeight*(getNegWeight()*(distQuery[i]/pSum)-(1-getNegWeight())*(negDistQuery[i]/nSum) )+
                    (1-expWeight)*ind.termCount(i)/ind.termCount();
        }

        lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);
        for (i=1; i<=numTerms; i++) {
            if (distQuery[i] > 0) {
                lmCounter.incCount(i, distQuery[i]);
            }
        }

        //cout<<"sum: "<<lmCounter.sum()<<endl;

        lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(lmCounter, ind.termLexiconID());
        origRep.interpolateWith(*fblm, 0.0, qryParam.fbTermCount,
                                qryParam.fbPrSumTh, 0.0);
        delete fblm;
        delete dCounter;
        delete nCounter;
        delete[] distQuery;
        delete[] negDistQuery;
    }
}
void lemur::retrieval::RetMethod::computeRM3FBModel(QueryModel &origRep,
                                                    const DocIDSet &relDocs)
{
    // Write Your own RM3 right here
}


// out: w.weight = P(w|Q)
// P(w|Q) = k P(w) P(Q|w)
// P(Q|w) = PROD_q P(q|w)
// P(q|w) = SUM_d P(q|d) P(w|d) p(d) / p(w)
// P(w) = SUM_d P(w|d) p(d)
// Promote this to some include somewhere...
struct termProb  {
    TERMID_T id; // TERM_ID
    double prob; // a*tf(w,d)/|d| +(1-a)*tf(w,C)/|C|
};

void lemur::retrieval::RetMethod::computeRM2FBModel(QueryModel &origRep,
                                                    const DocIDSet &relDocs) {
    COUNT_T numTerms = ind.termCountUnique();
    COUNT_T termCount = ind.termCount();
    double expWeight = qryParam.fbCoeff;

    // RelDocUnigramCounter computes P(w)=SUM(D){P(w|D)*P(D|Q)} for each w
    // P(w) = SUM_d P(w|d) p(d)
    lemur::langmod::RelDocUnigramCounter *dCounter = new lemur::langmod::RelDocUnigramCounter(relDocs, ind);

    double *distQuery = new double[numTerms+1];
    COUNT_T numDocs = ind.docCount();
    vector<termProb> **tProbs = new vector<termProb> *[numDocs + 1];

    int i;
    for (i=1; i<=numTerms;i++)
        distQuery[i] = 0.0;
    for (i = 1; i <= numDocs; i++) {
        tProbs[i] = NULL;
    }

    // Put these in a faster structure.
    vector <TERMID_T> qTerms; // TERM_ID
    origRep.startIteration();
    while (origRep.hasMore()) {
        QueryTerm *qt = origRep.nextTerm();
        qTerms.push_back(qt->id());
        delete(qt);
    }
    COUNT_T numQTerms = qTerms.size();
    dCounter->startIteration();
    while (dCounter->hasMore()) {
        int wd; // dmf fixme
        double P_w;
        double P_qw=0;
        double P_Q_w = 1.0;
        // P(q|w) = SUM_d P(q|d) P(w|d) p(d)
        dCounter->nextCount(wd, P_w);
        for (int j = 0; j < numQTerms; j++) {
            TERMID_T qtID = qTerms[j]; // TERM_ID
            relDocs.startIteration();
            while (relDocs.hasMore()) {
                int docID;
                double P_d, P_w_d, P_q_d;
                double dlength;
                relDocs.nextIDInfo(docID, P_d);
                dlength  = (double)ind.docLength(docID);
                if (tProbs[docID] == NULL) {
                    vector<termProb> * pList = new vector<termProb>;
                    TermInfoList *tList = ind.termInfoList(docID);
                    TermInfo *t;
                    tList->startIteration();
                    while (tList->hasMore()) {
                        t = tList->nextEntry();
                        termProb prob;
                        prob.id = t->termID();
                        prob.prob = expWeight*t->count()/dlength+
                                (1-expWeight)*ind.termCount(t->termID())/termCount;
                        pList->push_back(prob);
                    }
                    delete(tList);
                    tProbs[docID] = pList;
                }
                vector<termProb> * pList = tProbs[docID];
                P_w_d=0;
                P_q_d=0;
                for (int i = 0; i < pList->size(); i++) {
                    // p(q|d)= a*tf(q,d)/|d|+(1-a)*tf(q,C)/|C|
                    if((*pList)[i].id == qtID)
                        P_q_d = (*pList)[i].prob;

                    // p(w|d)= a*tf(w,d)/|d|+(1-a)*tf(w,C)/|C|
                    if((*pList)[i].id == wd)
                        P_w_d = (*pList)[i].prob;
                    if(P_q_d && P_w_d)
                        break;
                }
                P_qw += P_d*P_w_d*P_q_d;
            }
            // P(Q|w) = PROD_q P(q|w) / p(w)
            P_Q_w *= P_qw/P_w;
        }
        // P(w|Q) = k P(w) P(Q|w), k=1
        distQuery[wd] =P_w*P_Q_w;
    }

    lemur::utility::ArrayCounter<double> lmCounter(numTerms+1);
    for (i=1; i<=numTerms; i++) {
        if (distQuery[i] > 0) {
            lmCounter.incCount(i, distQuery[i]);
        }
    }
    lemur::langmod::MLUnigramLM *fblm = new lemur::langmod::MLUnigramLM(lmCounter, ind.termLexiconID());
    origRep.interpolateWith(*fblm, 0.0, qryParam.fbTermCount,
                            qryParam.fbPrSumTh, 0.0);
    delete fblm;
    delete dCounter;
    for (i = 1; i <= numDocs; i++) {
        delete(tProbs[i]);
    }
    delete[](tProbs);
    delete[] distQuery;
}

void lemur::retrieval::RetMethod::computeRM4FBModel(QueryModel &origRep,
                                                    const DocIDSet &relDocs)
{
    cout<<"haha";
    // Write Your own RM4 right here
}

void lemur::retrieval::RetMethod::computeKUNFBModel(QueryModel &origRep,
                                                    const DocIDSet &relDocs,const DocIDSet &nonRelDocs)
{
    COUNT_T numTerms = ind.termCountUnique();


}



