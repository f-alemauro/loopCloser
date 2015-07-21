#include "Loop_Closer.h"

using namespace std;
using namespace cv;
using namespace boost::filesystem;

int avgImg_rows = 0;            //number of features to be found in an average image
size_t totalNumbDescr=0;        //number of features found in all images
float totalDistance=0;			//how far are new descriptors from visual words?

bool pair_comparator (const pair<int,int> &l, const pair<int,int> &r);

map<int,vector<int> > Loop_Closer::recalcNewInvIndex( vector<int>::iterator curImgBow_beg, vector<int>::iterator curImgBow_end, int img_number) const {
	map<int,vector<int> > localInvIndex(newInvIndex);
	pair<int, vector<int> > inv_ind_pair;
	pair<map<int, vector<int> >::iterator, bool> inv_ind_write;
	int iteration_n=0;
	for(;curImgBow_beg!=curImgBow_end;++curImgBow_beg){
		if(*curImgBow_beg){
			vector<int> initial;
			initial.push_back(img_number);
			inv_ind_pair = make_pair(iteration_n, initial);
			inv_ind_write = localInvIndex.insert(inv_ind_pair);
			if(!inv_ind_write.second){
				inv_ind_write.first->second.push_back(img_number);
			}
		}
		++iteration_n;
	}
	return localInvIndex;
}

vector<float> Loop_Closer::calcImg_tfIdf(vector<int>::const_iterator bow_beg, vector<int>::const_iterator bow_end, const int n_features_img )const{

	vector<float> curImg_w_log;
	if (idf.rows==0){
		cout<<"!!!! - idf matrix is empty - !!!!"<<endl;
	}

	curImg_w_log.assign(bow_beg, bow_end);

	vector<float>::iterator  w_log_itr = curImg_w_log.begin(),
			w_log_itr_fal = curImg_w_log.end();
	int iteration = 0;

	for(;w_log_itr!=w_log_itr_fal;++w_log_itr){
		MatConstIterator_<float> temp_iter = idf.begin<float>()+iteration;
		if(*w_log_itr!=0) {														// if number of occurrences != 0
			//*w_log_itr = 1+log((*w_log_itr)/curImgFeatures.size());
			*w_log_itr = (*w_log_itr)/n_features_img;							// calculating tf
			*w_log_itr *=  *temp_iter;											// calculating tf*idf
		}
		++iteration;
	}

	return curImg_w_log;

}


map<int,float> Loop_Closer::getSimilarityCoefficients ( vector<int>::const_iterator bow_iterat,
		vector<int>::const_iterator bow_iterat_fal,
		vector<float>::const_iterator  itr_curImg_w_log,
		vector<float>::const_iterator  itr_curImg_w_log_fal,
		int neighbor,
		int img_n) const{
	map<int,float> sim_cfs;
	float sum_w = accumulate (itr_curImg_w_log,itr_curImg_w_log_fal,0);

	for(int i=-1; i!=img_n;++i){
		sim_cfs[i]-=sum_w;
	}
	int vw_number = 0;
	for(;bow_iterat!=bow_iterat_fal;++bow_iterat){
		if(*bow_iterat!=0){									// for each word found in the current image ....
			map<int, vector<int> >::const_iterator map_it = newInvIndex.find(vw_number);	// find other documents with this vw
			if(map_it!=newInvIndex.end()){
				vector<int>::const_iterator ii_iter = map_it->second.begin(), ii_iter_fal = map_it->second.end();
				for(;ii_iter!=ii_iter_fal;++ii_iter){										// for each image comprising this word
					vector<float>::const_iterator grab_iter = itr_curImg_w_log + vw_number;
					sim_cfs[*ii_iter] += 2*(*grab_iter);
				}
			}
		}
		++vw_number;
	}
	// to be turned on for avoiding closing the loop with neighbors
	for(int i=0; i!=neighbor;++i){
		sim_cfs.erase(img_n - i);
	}
	return sim_cfs;
}

vector<int> Loop_Closer::takeAvgImage(const int vw_n, const int avgImg_rows, int &n_descr_avgImg) const {

	vector<int> avgImgSow (vw_n, 0);                // preparing bow template
	vector<pair<int,int> > common_words;

	int allDescriptors = 0;

	//--------generate a vector of word-occur_times pairs------------------
	map<int, int>::const_iterator  map_it = vw_times_seen.begin(),
			map_it_fal = vw_times_seen.end();

	for(; map_it!=map_it_fal; ++map_it){
		common_words.push_back(*map_it);
	}


	//---- most frequent vw must be in the beginning ----------------------
	sort (common_words.begin(), common_words.end(), pair_comparator);


	//-------extracting first 'avgImg_rows' elements from the vector-------
	vector<pair<int,int> >::iterator itr_vw = common_words.begin(),
			itr_vw_fal = common_words.begin()+ avgImg_rows;

	if(itr_vw_fal>common_words.end()) itr_vw_fal = common_words.end();        // in some cases common_words.size() < avgImg_rows

	for (; itr_vw!=itr_vw_fal; ++itr_vw){
		//avgImgSow[itr_vw->first] = 1;                    // <-- sow representation
		avgImgSow[itr_vw->first] = itr_vw->second;        // <-- bow representation
		allDescriptors+=itr_vw->second;
	}

	n_descr_avgImg = allDescriptors;
	cout<<"average image calculated with "<< avgImg_rows<<" non-zero elements"<<endl;

	return avgImgSow;
}

map<int,float> Loop_Closer::getLikelihoodCoef (const map<int,float>::const_iterator sim_beg, const map<int,float>::const_iterator sim_fal) const {
	map<int,float> lkl;
	map<int,float>::const_iterator sim_first(sim_beg);
	pair<int,float> lkl_pair;
	float sum_sim_coef=0, n_coef=0, mean=0, deviation=0, val_mean_sq=0;
	//calculating mean
	for(;sim_first!=sim_fal;++sim_first){
		sum_sim_coef+=sim_first->second;
		++n_coef;
	}
	mean = sum_sim_coef/n_coef;
	// calculating standard deviation
	sim_first = sim_beg;
	for(;sim_first!=sim_fal;++sim_first){
		val_mean_sq+=pow((sim_first->second - mean), 2);
	}
	deviation = sqrt(val_mean_sq/n_coef);
	// making all vals below the threshold equal to 0
	sim_first = sim_beg;
	for(;sim_first!=sim_fal;++sim_first){
		lkl_pair = *sim_first;
		if(lkl_pair.second < (mean+deviation) ) lkl_pair.second = 1;
		else lkl_pair.second=lkl_pair.second/mean;            // <---- normalizing likelihood by 1/mean
		lkl.insert(lkl_pair);
	}
	return lkl;
}

Mat Loop_Closer::calcIdfIset(vector<int>::const_iterator bow_beg, const int total_n_vw) const {
	// go to inverted index and check the number of images comprising each word
	Mat idf_iset_dev(total_n_vw, 1, CV_32FC1, 1);	// a Mat-vector to store w, 1s won't be changed and will turn to 0 after log operation
	Mat idf_log;
	MatIterator_<float> iter_idf_dev = idf_iset_dev.begin<float>(), iter_idf_dev_fal = idf_iset_dev.end<float>();
	int n_run = 0;
	for(;iter_idf_dev!=iter_idf_dev_fal;++iter_idf_dev, ++bow_beg){				// for each position... (total number is equal to total number of vw)
		if (*bow_beg!=0)  {		// we wont need idf for words, idf of which  is equal to 0
			float n_img_with_vw=0;
			map<int, vector<int> >::const_iterator iter_idf_cur = newInvIndex.find(n_run);                // is the word already in the inverted index?
			if (iter_idf_cur == newInvIndex.end())
				n_img_with_vw = newKeyPointsByImage.size();            // if the word was not seen before make the idf = 0 is it a good idea??
			else
				n_img_with_vw = iter_idf_cur->second.size();
			*iter_idf_dev = (newKeyPointsByImage.size()) / n_img_with_vw;
			++n_run;
		}
	}

	log(idf_iset_dev, idf_log);        // calculate log of each matrix element
	return idf_log;
}

bool pair_comparator (const pair<int,int> &l, const pair<int,int> &r){
	return l.second > r.second;
}


std::vector<std::vector<float> >Loop_Closer::calcProbabCamera(Vocabulary &VocabularyObject, flann::GenericIndex<cv::flann::L2<float> >  &flannIndObj, const unsigned neighbor, const float lc_threshold,const unsigned subset_range,const float gmtr_threshold, Mat &img_current, unsigned int img_n) throw (runtime_error){
	detectedLoopClosures.clear();
	cout<<"Processing image number "<<img_n<<endl;
	Mat curImgFlannLabels, curImgFlannDist, curImgDescriptors;
	vector<KeyPoint> curImgFeatures;

	//******* get sow_representation of a new image **********
	cout<<"Starting sow_bow_computation..."<<endl;
	vector<int> curImg_bow = VocabularyObject.getSowBowRepr_img(img_current, 0, flannIndObj, curImgFeatures, curImgDescriptors, curImgFlannLabels, curImgFlannDist);
	if (*curImg_bow.begin() == -1) {
		throw runtime_error("Error in BoW representation");	// just ignoring should work in this case
	}

	bowImgRepresByImage.push_back(curImg_bow);
	vwImgRepresByImage.push_back(curImgFlannLabels);			// vector of words each image consists of
	newKeyPointsByImage.push_back(curImgFeatures);
	newDescriptorsByImage.push_back(curImgDescriptors);
	//--------increment total distance (denk mal)-------
	MatConstIterator_<float> dist_it = curImgFlannDist.begin<float>(), dist_it_fal = curImgFlannDist.end<float>();
	for(;dist_it!=dist_it_fal; ++dist_it){
		totalDistance+=*dist_it;
	}


	//--------------------------------------------------
	totalNumbDescr+=curImgFeatures.size();
	//************** end of get bow representation **************
	cout<<"Starting new Inverted Index Computing..."<<endl;
	//-----------recompute inverted index------------------
	vector<int>::iterator itr_sow = curImg_bow.begin(), itr_sow_fal = curImg_bow.end();
	newInvIndex = recalcNewInvIndex( itr_sow, itr_sow_fal, img_n);
	cout<<"New Inverted Index Computing ended!"<<endl;
	cout<<"INVIND:::"<<newInvIndex.size()<<endl;

	// ======== updating vw_times_seen map ============
	MatConstIterator_<int> itr_labels = curImgFlannLabels.begin<int>(), itr_labels_fal = curImgFlannLabels.end<int>();
	for(; itr_labels!=itr_labels_fal; ++itr_labels){
		++vw_times_seen[*itr_labels];
	}
	//==================================================


	cout<<"Starting tf-idf weight computation..."<<endl;
	idf = calcIdfIset(curImg_bow.begin(), curImg_bow.size()); //COSA SERVE????
	pair<int, vector<float> > tfIdfByImage_pair;

	vector<float> curImg_w_log = calcImg_tfIdf(curImg_bow.begin(), curImg_bow.end(), curImgFeatures.size());
	if (curImg_w_log.size()==0){
		throw runtime_error ("Error occurred while forming curImg_w_log");
	}
	tfIdfByImage_pair = make_pair(img_n, curImg_w_log);
	tfIdfByImage.insert(tfIdfByImage_pair);
	//-----------------------------------------------------------
	cout<<"tf-idf weight computation ended"<<endl;

	if (img_n > neighbor) {
		//++++++++++++++++++ average image calculation +++++++++++++++++++++++++++
		if (img_n<100 || (img_n-neighbor)%100==0){				// is it enough??
			cout<<"Calculating new average image..."<<endl;
			//-----------deleting '-1's from the inverse index-----------------------------
			map<int, vector<int> >::iterator ii_map_itr = newInvIndex.begin(), ii_map_itr_fal = newInvIndex.end();
			for(;ii_map_itr!=ii_map_itr_fal; ++ii_map_itr){                                //for each key
				vector<int>::iterator inner_runner = ii_map_itr->second.begin(), inner_runner_fal = ii_map_itr->second.end();
				for(;inner_runner!=inner_runner_fal; ++inner_runner){                        // for each element of the member vector
					if((*inner_runner)==-1)
						ii_map_itr->second.erase(inner_runner);        // delete if it is equal to -1
				}
			}
			//------------------------------------------------------------------------------

			int n_descr_avgImg=0;
			avgImg_rows = totalNumbDescr/newKeyPointsByImage.size();
			//-------trying to get average image from current ones
			bowAvgImage = takeAvgImage(curImg_bow.size(),avgImg_rows, n_descr_avgImg);
			vector<int>::iterator itr_avg_bow = bowAvgImage.begin(), itr_avg_bow_fal = bowAvgImage.end();
			newInvIndex = recalcNewInvIndex( itr_avg_bow, itr_avg_bow_fal, -1);

		}
		//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



		//******** calc. similarity map ******************************
		cout<<"Starting similarity map computation..."<<endl;
		map<int,float> sim_cfs = getSimilarityCoefficients(curImg_bow.begin(),curImg_bow.end(), curImg_w_log.begin(), curImg_w_log.end(), neighbor, img_n );
		map<int,float> likelihood = getLikelihoodCoef(sim_cfs.begin(), sim_cfs.end());

		lc_probability.push_back(likelihood);

		int fal_not_neighbor = img_n-neighbor;
		if (fal_not_neighbor==1){
			vector<float> probability(1,1);
			pair<int, vector<float> > clp_pair;
			clp_pair = make_pair(neighbor,probability);
			closureProbab.insert(clp_pair);
		}
		//----------------------------------------------------------------
		vector<float> curImg_pos_probab;
		float p_lclosing=0;
		//double aposteriori_start=(double)getTickCount();

		for(int to_pos = -1; to_pos!=fal_not_neighbor;++to_pos){
			float p_lclosing_with_to_pos=0;

			if (to_pos==-1){
				p_lclosing_with_to_pos=0.9*closureProbab[img_n-1].at(to_pos+1) + 0.1*(1-closureProbab[img_n-1].at(to_pos+1));
			}
			else {
				int a = to_pos-2 >=0 ? to_pos-2 : 0;
				int b = to_pos+3 < fal_not_neighbor-1 ? to_pos+3 : fal_not_neighbor-1;
				for(int from_pos=a; from_pos!=b;++from_pos){
					float trans=0;
					if(abs(to_pos-from_pos) == 0) trans=0.115;
					if(abs(to_pos-from_pos) == 1) trans=0.225;
					if(abs(to_pos-from_pos) == 2) trans=0.1675;

					float p_being_in_from_pos = closureProbab[img_n-1].at(from_pos+1);
					p_lclosing_with_to_pos+= p_being_in_from_pos * trans;
				}

				p_lclosing_with_to_pos+=(0.1/(img_n-neighbor))*closureProbab[img_n-1].at(0);

			}

			float fp_cur_pos  = p_lclosing_with_to_pos * likelihood[to_pos+1];
			curImg_pos_probab.push_back(fp_cur_pos);
			p_lclosing+=fp_cur_pos;
		}


		cout<<"   +++p_lclosing: "<<p_lclosing<<endl;
		vector<float>::iterator lc_iter = curImg_pos_probab.begin(),lc_iter_fal = curImg_pos_probab.end();
		deque<float> subset;
		float omg =0;
		int position_number=0;
		vector<pair<int,float> > probableClosureCandidates;
		vector<float> subset_sums;
		float sum = 0;


		for(;lc_iter!=lc_iter_fal; ++lc_iter){
			if(lc_iter== curImg_pos_probab.begin()){
				*lc_iter = *lc_iter / p_lclosing;
			}
			if(lc_iter!= curImg_pos_probab.begin()){
				subset.push_back(*lc_iter);
				sum+=*lc_iter;
				if ( subset.size() > subset_range ) {
					sum-=subset.front();
					subset.pop_front();
				}
				subset_sums.push_back(sum);
				*lc_iter = *lc_iter / p_lclosing;
				if (sum>=lc_threshold){
					pair<int,float> matchProbab;
					int simImg = position_number - subset.size()/2+1;        // some trouble is possible in the very beginning
					cout<<"Threshold requirements met by image "<<simImg<<endl;
					matchProbab=make_pair(simImg, sum);
					cout<<"Putting pair in the list ... "<<endl;
					probableClosureCandidates.push_back(matchProbab);
				}
			}
			omg+=*lc_iter;
			++position_number;
		}


		closureProbabSubset.insert(make_pair(img_n, subset_sums));
		cout<<"+++omg: "<<omg<<endl;
		cout<<"lc detected: "<<detectedLoopClosures.size()<<endl;
		if(probableClosureCandidates.size()!=0){
			string matcherType("FlannBased");
			Ptr<DescriptorMatcher> descriptorMatcher;
			descriptorMatcher = DescriptorMatcher::create( matcherType );
			cout<<"  descriptor matcher was created with type "<<matcherType<<endl;
			Mat trainDescriptors = newDescriptorsByImage.at(img_n);
			vector<Mat> bzzz;
			bzzz.push_back(trainDescriptors);
			descriptorMatcher->add( bzzz );
			cout<<"Descriptor matcher was given image "<<img_n<<endl;
			vector<pair<int,float> >::const_iterator candidate_it = probableClosureCandidates.begin(), candidate_it_fal = probableClosureCandidates.end();
			for(;candidate_it!=candidate_it_fal;++candidate_it){
				vector<float> cur_closure;
				Mat queryDescriptors = newDescriptorsByImage.at(candidate_it->first);
				vector<DMatch> indices;
				descriptorMatcher->match( queryDescriptors, indices );
				cout<<"  matching done for possible candidate "<<candidate_it->first<<endl;
				//++++++++++finding out which points match properly+++++++++++++++++++++
				vector<DMatch>::const_iterator 	ind_iter = indices.begin(),
						ind_iter_fal = indices.end();
				vector<pair<int,int> > compatPoints;
				pair<int, int> compatPair;
				for(;ind_iter!=ind_iter_fal;++ind_iter){
					compatPair = make_pair(ind_iter->trainIdx, ind_iter->queryIdx);
					compatPoints.push_back(compatPair);
				}
				//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

				//------------------generating two query matrices-------------------------------
				Mat points1(compatPoints.size(), 2, CV_32FC1),
						points2(compatPoints.size(), 2, CV_32FC1);

				vector<pair<int, int> >::const_iterator   kpMatch_it = compatPoints.begin(),
						kpMatch_it_fal = compatPoints.end();

				vector<vector<KeyPoint> >::const_iterator  vvp_iter_img_n = newKeyPointsByImage.begin()+img_n,
						vvp_iter_simImg = newKeyPointsByImage.begin()+candidate_it->first;

				MatIterator_<float> pt1_it = points1.begin<float>(), pt2_it = points2.begin<float>();

				//making two matrices with coordinates of matching features
				for(;kpMatch_it!=kpMatch_it_fal;++kpMatch_it){
					*pt1_it = vvp_iter_img_n->at(kpMatch_it->first).pt.x;
					++pt1_it;
					*pt1_it = vvp_iter_img_n->at(kpMatch_it->first).pt.y;
					++pt1_it;
					*pt2_it = vvp_iter_simImg->at(kpMatch_it->second).pt.x;
					++pt2_it;
					*pt2_it = vvp_iter_simImg->at(kpMatch_it->second).pt.y;
					++pt2_it;
				}
				//-----------------------------------------------------------------------------------

				vector<uchar> status;
				//it might be useful to set all the parameters manually
				findFundamentalMat(points1, points2, status, FM_RANSAC, 3., 0.9);			// what corresponds to '3' method&

				vector<uchar>::iterator stat_itr = status.begin(),
						stat_itr_fal = status.end();
				int calc_nonzero=0;
				for(; stat_itr!=stat_itr_fal;++stat_itr ){
					if (*stat_itr!=0){					//and did that '3' method work with outlyers?
						++calc_nonzero;
					}
				}
				float ratio = 0;
				if(compatPoints.size()!=0){
					ratio = static_cast<float>(calc_nonzero)/ static_cast<float>(compatPoints.size());
				}
				if (ratio>=gmtr_threshold) {

					cur_closure.push_back(static_cast<float>(img_n));
					cur_closure.push_back(static_cast<float>(candidate_it->first));
					cur_closure.push_back(candidate_it->second);
					cur_closure.push_back(ratio);
					detectedLoopClosures.push_back(cur_closure);

				}
				//************************************************************************************
				pair<int, vector<float> > fp_pair;
				fp_pair = make_pair(img_n, curImg_pos_probab);
				closureProbab.insert(fp_pair);
			}
		}
		pair<int, vector<float> > fp_pair;
		fp_pair = make_pair(img_n, curImg_pos_probab);
		closureProbab.insert(fp_pair);

	}
	return detectedLoopClosures;
}


