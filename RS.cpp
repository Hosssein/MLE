#ifndef SMTH
#define SMTH

#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <utility>
#include <sstream>
#include <DocStream.hpp>
#include <BasicDocStream.hpp>
#include "IndexManager.hpp"
#include "ResultFile.hpp"
//#include "DocUnigramCounter.hpp"
#include "RetMethod.h"
#include "QueryDocument.hpp"
#include <sstream>

#include "Parameters.h"

#include <FreqVector.hpp>


using namespace lemur::api;
using namespace lemur::langmod;
using namespace lemur::parse;
using namespace lemur::retrieval;
using namespace std;

void loadJudgment();
void computeRSMethods(Index *);
void MonoKLModel(Index* ind);
vector<int> queryDocList(Index* ind,TextQueryRep *textQR);


template <typename T>
string numToStr(T number)
{
    ostringstream s;
    s << number;
    return s.str();
}

extern double startThresholdHM , endThresholdHM , intervalThresholdHM ;
extern int WHO;// 0--> server , 1-->Mozhdeh, 2-->AP, other-->Hossein
extern string outputFileNameHM;
extern string resultFileNameHM;
extern int feedbackMode;
extern double startNegWeight,endNegWeight , negWeightInterval;
extern double startNegMu, endNegMu, NegMuInterval;
extern double startDelta, endDelta, deltaInterval;
extern int RSMethodHM;
extern int negGenModeHM;
extern double smoothJMInterval1,smoothJMInterval2;

extern int updatingThresholdMode;


map<string , vector<string> >queryRelDocsMap;
map<string ,vector<string> > queryRelTrainedDocsMap;
string judgmentPath,indexPath,queryPath;
string resultPath = "";

vector<double>judgScores;
vector<double>judgThrs;
vector<bool>judgRelNonRel;//true: rel
double expectedRatioRel =0.0;

double ralpha =1 , rbeta = 1 , rgama = 1;

#define DATASET 0 //0-->infile, 1-->ohsu


#define RETMODE RSMethodHM//LM(0) ,RS(1), NegKLQTE(2),NegKL(3)
#define NEGMODE negGenModeHM//coll(0) ,NonRel(1)
#define FBMODE feedbackMode//NoFB(0),NonRel(1),Normal(2),Mixture(3)
#define UPDTHRMODE 1//updatingThresholdMode//No(0),Linear(1) ,Diff(2)

//void initQueryVec(Index* ,string qid  , lemur::api::TextQueryRep *);

//const int dim = 148000;
vector<double>hquery;
vector<double>hqueryZero;
vector<double>hrel;
vector<double>hnonRel;
vector<double>diagonal;

double sumOfRelScores = 0 ,sumOfNonRelScores = 0;

bool endFilteringForQuery = false;

int main(int argc, char * argv[])
{
    readParams(string(argv[1]));
    cout<< "reading param file: "<<argv[1]<<endl;
    switch (WHO)
    {
    case 0:
        if(DATASET == 0)//infile
        {
            judgmentPath = "/home/iis/Desktop/Edu/thesis/Data/INFILE/qrels_en";
            indexPath= "/home/iis/Desktop/Edu/thesis/index/infile/en/index.key";
            queryPath = "/home/iis/Desktop/Edu/thesis/Data/INFILE/q_en_titleKeyword_en.stemmed.xml";
        }else if(DATASET == 1)//ohsu
        {
            //judgmentPath = "/home/iis/Desktop/Edu/thesis/Data/ohsumed/trec9-train/qrels.ohsu.adapt.87";
            judgmentPath = "/home/iis/Desktop/Edu/thesis/Data/ohsumed/trec9-test/qrels.ohsu.88-91";
            //indexPath= "/home/iis/Desktop/Edu/thesis/index/ohsumed/ohsu/index/index.key";
            indexPath= "/home/iis/Desktop/Edu/thesis/index/ohsumed/ohsu/testIndex/index/index.key";
            queryPath = "/home/iis/Desktop/Edu/thesis/Data/ohsumed/trec9-train/trec9-train_output/stemmed_ohsu_query.txt";
        }
        break;
    case 1:
        judgmentPath = "/home/mozhdeh/Desktop/INFILE/hosein-data/qrels_en";
        indexPath = "/home/mozhdeh/Desktop/INFILE/javid-index/index.key";
        queryPath = "/home/mozhdeh/Desktop/INFILE/hosein-data/q_en_titleKeyword_en.stemmed.xml";
        break;
        //case 2:
        //    judgmentPath ="/home/mozhdeh/Desktop/AP/Data/jud-ap.txt";
        //    indexPath = "/home/mozhdeh/Desktop/AP/index/index.key";
        //   queryPath = "/home/mozhdeh/Desktop/AP/Data/topics.stemmed.xml";
        //   break;
    default:
        if(DATASET == 0)//infile
        {
            judgmentPath = "/home/hossein/Desktop/IIS/Lemur/DataSets/Infile/Data/qrels_en";
            indexPath = "/home/hossein/Desktop/IIS/Lemur/DataSets/Infile/Index/en_Stemmed_withoutSW/index.key";
            queryPath = "/home/hossein/Desktop/IIS/Lemur/DataSets/Infile/Data/q_en_titleKeyword_en.stemmed.xml";//????????
        }else if(DATASET == 1)//ohsu
        {
            judgmentPath = "/home/hossein/Desktop/lemur/DataSets/Ohsumed/Data/trec9-train/qrels.ohsu.adapt.87";
            indexPath = "/home/hossein/Desktop/lemur/DataSets/Ohsumed/Index/trec9-train/index.key";
            queryPath = "/home/hossein/Desktop/lemur/DataSets/Ohsumed/Data/trec9-train/stemmed_ohsu_query.txt";

        }

        break;
    }

    Index *ind = IndexManager::openIndex(indexPath);// Your own path to index

    const int dim = ind->termCountUnique()+1;//one for OOV
    diagonal.assign(dim , 0.0);
    for(int i = 1 ; i <= ind->termCountUnique() ; i++)
    {
        //cerr<<(double)ind->docCount(i)<<" ";
        diagonal[i] = log10( (ind->docCount() + 1.0) / (double)ind->docCount(i) );
    }

    loadJudgment();
    expectedRatioRel /= ind->docCount();
    computeRSMethods(ind);

    //MonoKLModel(ind);

}



void computeRSMethods(Index* ind)
{
    DocStream *qs = new BasicDocStream(queryPath); // Your own path to topics
    ArrayAccumulator accumulator(ind->docCount());
    RetMethod *myMethod = new RetMethod(*ind,"",accumulator);


    string outFilename;
    string thrUpdatingName = "SD";//MEAN1,MEAN2
    if(DATASET == 0)
        outFilename =outputFileNameHM+"_infile_MLE_"+thrUpdatingName;
    else if (DATASET == 1)
        outFilename =outputFileNameHM+"_ohsu";

    ofstream out(outFilename.c_str());


    cout<< "RSMethod: "<<RSMethodHM<<" NegGenMode: "<<negGenModeHM<<" feedbackMode: "<<feedbackMode<<" updatingThrMode: "<<updatingThresholdMode<<"\n";
    cout<< "RSMethod: "<<RETMODE<<" NegGenMode: "<<NEGMODE<<" feedbackMode: "<<FBMODE<<" updatingThrMode: "<<UPDTHRMODE<<"\n";
    cout<<"outfile: "<<outFilename<<endl;
    double start_thresh =startThresholdHM, end_thresh= endThresholdHM;


    for (double thresh = start_thresh ; thresh<=end_thresh ; thresh += intervalThresholdHM)
    {
        myMethod->setThreshold(thresh);

        resultPath = thrUpdatingName +"_"+resultFileNameHM +"_thr:"+numToStr( myMethod->getThreshold() )+ ".res" ;//+numToStr( myMethod->getThreshold() )+"_c1:"+numToStr(c1)+"_c2:"+numToStr(c2)+"_#showNonRel:"+numToStr(numOfShownNonRel)+"_#notShownDoc:"+numToStr(numOfnotShownDoc)+".res";


        IndexedRealVector results;

        out<<"threshold: "<<myMethod->getThreshold()<<endl;
        qs->startDocIteration();
        TextQuery *q;

        ofstream result(resultPath.c_str());
        ResultFile resultFile(1);
        resultFile.openForWrite(result,*ind);

        double relRetCounter = 0 , retCounter = 0 , relCounter = 0;
        vector<double> queriesPrecision,queriesRecall;

        const int dim = ind->termCountUnique()+1;//one for OOV
        while(qs->hasMore())
        {            
            myMethod->setThreshold(thresh);

            endFilteringForQuery = false;
            bool recievedRel = false;
            hqueryZero.assign(dim , 0.0);
            hquery.assign(dim , 0.0);
            hrel.assign(dim , 0.0);
            hnonRel.assign(dim , 0.0);



            sumOfNonRelScores = 0;
            sumOfRelScores = 0;
            judgRelNonRel.clear();
            judgScores.clear();
            judgThrs.clear();

            int numberOfNotShownDocs = 0,numberOfShownNonRelDocs = 0;

            int nn = 0 , rr =1;

            vector<int> relJudgDocs,nonRelJudgDocs;
            results.clear();


            Document* d = qs->nextDoc();
            q = new TextQuery(*d);
            QueryRep *qr = myMethod->computeQueryRep(*q);
            cout<<"qid: "<<q->id()<<endl;

            myMethod->initQueryVec( q->id() , (TextQueryRep *)(qr) );/*********************************/
            //return;


            bool newNonRel = false , newRel = false;
            vector<string> relDocs;

            if( queryRelDocsMap.find(q->id()) != queryRelDocsMap.end() )//find it!
                relDocs = queryRelDocsMap[q->id()];
            else
            {
                cerr<<"*******have no rel judge docs**********\n";
                continue;
            }

            //for(int docID = 1 ; docID < ind->docCount() ; docID++){ //compute for all doc
            vector <int> docids = queryDocList(ind,((TextQueryRep *)(qr)));

            for(int i = 0 ; i<docids.size(); i++) //compute for docs which have queryTerm
            {
                int docID = docids[i];

                float sim = myMethod->computeProfDocSim(((TextQueryRep *)(qr)) ,docID, relJudgDocs , nonRelJudgDocs , newNonRel,newRel);

                if(sim >=  myMethod->getThreshold() )
                {

                    judgScores.push_back(sim);
                    judgThrs.push_back(myMethod->getThreshold());


                    numberOfNotShownDocs=0;
                    bool isRel = false;
                    const EXDOCID_T did = ind->document(docID);
                    for(int ii = 0 ; ii < relDocs.size() ; ii++)
                    {
                        if(relDocs[ii] == did )
                        {
                            rr++;
                            sumOfRelScores += sim;
                            judgRelNonRel.push_back(true);
                            isRel = true;
                            newNonRel = false;
                            newRel = true;
                            relJudgDocs.push_back(docID);
                            //relSumScores+=sim;
                            //numberOfShownNonRelDocs = 0;
                            break;
                        }
                    }
                    if(!isRel)
                    {
                        nn++;
                        sumOfNonRelScores += sim;
                        judgRelNonRel.push_back(false);
                        nonRelJudgDocs.push_back(docID);
                        newNonRel = true;
                        newRel = false;
                        //nonRelSumScores+=sim;
                        numberOfShownNonRelDocs++;
                    }
                    results.PushValue(docID , sim);

                    if(results.size() > 200)
                    {
                        cout<<"BREAKKKKKKKKKK result size > 200\n";
                        break;
                    }

                    //if (results.size() % 15 == 0 )/************************************/
                    //parametersStimation(relJudgDocs , nonRelJudgDocs);

                    //if (results.size() % 15 == 0 )
                    if(rr % 6 == 0)
                    {
                        recievedRel = true;
                        rr =1;
                        myMethod->updateThreshold(*((TextQueryRep *)(qr)) ,relJudgDocs ,nonRelJudgDocs ,3 );
                        //myMethod->updateProfile(*((TextQueryRep *)(qr)), relJudgDocs , nonRelJudgDocs, isRel );
                    }
                    if(recievedRel)//FIX ME if the first 6 doc is not nonRel!!!!!!!!!!!!!
                        myMethod->updateProfile(*((TextQueryRep *)(qr)), relJudgDocs , nonRelJudgDocs, isRel );

                    if(endFilteringForQuery)
                        break;
                }
                else
                {
                    newNonRel = false;
                    newRel = false;
                    numberOfNotShownDocs++;
                }

            }//endfor docs

            results.Sort();
            resultFile.writeResults(q->id() ,&results,results.size());
            relRetCounter += relJudgDocs.size();
            retCounter += results.size();
            relCounter += relDocs.size();

            if(results.size() != 0)
            {
                queriesPrecision.push_back((double)relJudgDocs.size() / results.size());
                queriesRecall.push_back((double)relJudgDocs.size() / relDocs.size() );
            }else // have no suggestion for this query
            {
                queriesPrecision.push_back(0.0);
                queriesRecall.push_back(0.0);
            }


            //break;
            delete q;
            delete qr;

        }//end queries


        double avgPrec = 0.0 , avgRecall = 0.0;
        for(int i = 0 ; i < queriesPrecision.size() ; i++)
        {
            avgPrec+=queriesPrecision[i];
            avgRecall+= queriesRecall[i];
            out<<"Prec["<<i<<"] = "<<queriesPrecision[i]<<"\tRecall["<<i<<"] = "<<queriesRecall[i]<<endl;
        }
        avgPrec/=queriesPrecision.size();
        avgRecall/=queriesRecall.size();


        out<<"Avg Precision: "<<avgPrec<<endl;
        out<<"Avg Recall: "<<avgRecall<<endl;
        if(avgPrec == 0 || avgPrec == 0)
            out<<"F-measure: 0"<<endl<<endl;
        else
            out<<"F-measure: "<<(2*avgPrec*avgRecall)/(avgPrec+avgRecall)<<endl<<endl;

        double pp = relRetCounter/retCounter;
        double dd = relRetCounter/relCounter;
        out<<"rel_ret: "<<relRetCounter<<" ret: "<<retCounter<<" rels: "<<relCounter<<endl;
        out<<"old_Avg Precision: "<<pp<<endl;
        out<<"old_Avg Recall: "<<dd<<endl;
        if(pp == 0 || dd == 0)
            out<<"old_F-measure: 0"<<endl<<endl;
        else
            out<<"old_F-measure: "<<(2*pp*dd)/(pp+dd)<<endl<<endl;


#if RETMODE == 1 && FBMODE == 1
    }
#endif



}
//#endif
delete qs;
delete myMethod;
}

/*double ltuQueryWeighting(Index* ind ,  lemur::api::TextQueryRep *textQR )
{
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
        double T = log10((double)(ind->docCount()+1.0) / ind->docCount(qt->id() ) ) ;

        double avgUniqueTermPerDoc = ind->termCountUnique()/ ind->docCount();
        double U = 1.0 / (0.8 +0.2*( (double)( qset.size() ) / avgUniqueTermPerDoc ) );


        //hquery[qt->id()] += L*T*U;
        delete qt;

        return L*T*U;
    }

    return -1;
}*/

#if 0
void parametersStimation()
{
    //H=(u, delta, lambda, p)
    //not using congucate gradient descent

    double eps = 0.001 , V = 0.005;

    double pPrior = pow(expectedRatioRel,eps) * pow((1-expectedRatioRel) ,eps);
    double deltaPrior = exp((-V*V)/2*var);



    if(judgRelNonRel.size() != judgScores.size())
        cerr<<"whyyyyyyyyyyyyyyyy???\n\n";
    const double minScore = 0.3;//fix meeeeeeeeeee!

    double maxSigma = -10 , maxU=-10, maxDelta =-10,maxLambda =-10,maxP = -10;
    for(double u = 0.0 ; u < 1 ;u+=0.5)
        for(double delta = 0.0 ; delta < 1 ;delta+=0.5)
            for(double lambda = 0.0 ; lambda < 1 ;lambda+=0.5)
                for(double p = 0.0 ; p < 1 ;p+=0.5)
                {
                    double sig1= 0.0 , sig2 = 0.0;
                    for(int i = 0 ; i < judgRelNonRel.size() ; i++)
                    {
                        double g = gFunc(u,delta ,lambda, p, judgThrs[i]);
                        if(judgRelNonRel[i] == true)//Rel
                        {
                            sig1 += ( (judgScores[i]-u)*(judgScores[i]-u) )/(2*delta*delta);
                            sig2 += log( p/delta * g );

                        }else
                        {
                            sig1 += lambda*(judgScores[i] - minScore );
                            sig2 += log( ( (1-p)*lambda) / g);
                        }
                    }
                    if (sig1+sig2 > maxSigma)
                    {
                        maxSigma = sig1+sig2;
                        maxU = u;
                        maxDelta = delta;
                        maxLambda = lambda;
                        maxP = p;
                    }


                }
}
void distributionParams(double &hmeanN ,double &hvarN ,double &hmeanE, double &hvarE)
{
    double meanN = 0.0 , meanE = 0.0;
    int cc = 0;
    for(int i = 0 ; i < judgRelNonRel.size() ;i++ )
    {
        if(judgRelNonRel[i]== true)
        {
            cc++;
            meanN+=judgScores[i];
        }else
        {
            meanE +=judgScores[i];
        }
    }
    meanN /= cc;
    meanE /= (judgRelNonRel.size() - cc);

    double sig1 = 0.0 , sig2 = 0.0;
    for(int i = 0 ; i <  judgRelNonRel.size() ; i++)
    {
        if(judgRelNonRel[i] == true)
            sig1 += (judgScores[i] - meanN)*(judgScores[i] - meanN);
        else
            sig2 += (judgScores[i] - meanE)*(judgScores[i] - meanE);
    }
    hmeanN = meanN;
    hmeanE = meanE;
    hvarN = sig1 /cc;
    hvarE = sig2 /(judgRelNonRel.size() - cc);
}
double gFunc(double u,double delta,double lambda,double p, double tetha)
{
    return 1.0;
}
#endif
void loadJudgment()
{
    int judg,temp;
    string docName,id;

    ifstream infile;
    infile.open (judgmentPath.c_str());

    int trainCnt = 2;
    string line;
    while (getline(infile,line))
    {
        stringstream ss(line);
        if(DATASET == 0)//infile
        {
            ss >> id >> temp >> docName >> judg;
            if(judg == 1)
            {
                queryRelDocsMap[id].push_back(docName);
                //cerr<<id<<" "<<docName<<endl;


                expectedRatioRel +=1.0;
            }
        }else if(DATASET == 1)//ohsu
        {
            ss >> id >> docName;
            queryRelDocsMap[id].push_back(docName);

            expectedRatioRel +=1.0;
        }

    }

    map<string,vector<string> >::iterator cit;
    for ( cit = queryRelDocsMap.begin() ; cit != queryRelDocsMap.end() ; ++cit )
    {
        vector<string>temp;

        for(int i = 0 ; i < std::min(trainCnt , (int)cit->second.size() ) ; i++)
            temp.push_back(cit->second[i]) ;

        queryRelTrainedDocsMap.insert(make_pair<string, vector<string> >(cit->first , temp ) );
    }

    infile.close();


    //110,134,147 rel nadaran
    /*map<string , vector<string> >::iterator it;
    for(it = queryRelDocsMap.begin();it!= queryRelDocsMap.end() ; ++it)
        cerr<<it->first<<endl;*/

}
vector<int> queryDocList(Index* ind,TextQueryRep *textQR)
{
    vector<int> docids;
    set<int> docset;
    textQR->startIteration();
    while (textQR->hasMore()) {
        QueryTerm *qTerm = textQR->nextTerm();
        if(qTerm->id()==0){
            cerr<<"**********"<<endl;
            continue;
        }
        DocInfoList *dList = ind->docInfoList(qTerm->id());

        dList->startIteration();
        while (dList->hasMore()) {
            DocInfo *info = dList->nextEntry();
            DOCID_T id = info->docID();
            docset.insert(id);
        }
        delete dList;
        delete qTerm;
    }
    docids.assign(docset.begin(),docset.end());
    return docids;
}

void MonoKLModel(Index* ind){
    DocStream *qs = new BasicDocStream(queryPath.c_str()); // Your own path to topics
    ArrayAccumulator accumulator(ind->docCount());
    RetMethod *myMethod = new RetMethod(*ind,"",accumulator);
    IndexedRealVector results;
    qs->startDocIteration();
    TextQuery *q;

    ofstream result("res.my_ret_method");
    ResultFile resultFile(1);
    resultFile.openForWrite(result,*ind);
    PseudoFBDocs *fbDocs;
    while(qs->hasMore()){
        Document* d = qs->nextDoc();
        //d->startTermIteration(); // It is how to iterate over query terms
        //ofstream out ("QID.txt");
        //while(d->hasMore()){
        //	const Term* t = d->nextTerm();
        //	const char* q = t->spelling();
        //	int q_id = ind->term(q);
        //	out<<q_id<<endl;
        //}
        //out.close();
        q = new TextQuery(*d);
        QueryRep *qr = myMethod->computeQueryRep(*q);
        myMethod->scoreCollection(*qr,results);
        results.Sort();
        //fbDocs= new PseudoFBDocs(results,30,false);
        //myMethod->updateQuery(*qr,*fbDocs);
        //myMethod->scoreCollection(*qr,results);
        //results.Sort();
        resultFile.writeResults(q->id(),&results,results.size());
        cerr<<"qid "<<q->id()<<endl;
        break;
    }
}


#if 0
#include "pugixml.hpp"
using namespace pugi;
void ParseQuery(){
    ofstream out("topics.txt");
    xml_document doc;
    xml_parse_result result = doc.load_file("/home/hossein/Desktop/lemur/DataSets/Infile/Data/q_en.xml");// Your own path to original format of queries
    xml_node topics = doc.child("topics");
    for (xml_node_iterator topic = topics.begin(); topic != topics.end(); topic++){
        xml_node id = topic->child("identifier");
        xml_node title = topic->child("title");
        xml_node desc = topic->child("description");
        xml_node nar = topic->child("narrative");
        out<<"<DOC>"<<endl;
        out<<"<DOCNO>"<<id.first_child().value()<<"</DOCNO>"<<endl;
        out<<"<TEXT>"<<endl;
        out<<title.first_child().value()<<endl;
        out<<"</TEXT>"<<endl;
        out<<"</DOC>"<<endl;

    }
    printf("Query Parsed.\n");
}
#endif
#endif
