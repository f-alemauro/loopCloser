#include "Vocabulary.h"

bool comparator (const pair<int,int> &l, const pair<int,int> &r);

void Vocabulary::acquireImgNames(const std::string &dir_path, const std::string &fileExtension) throw(std::string){
	if(!boost::filesystem::exists(dir_path) || !boost::filesystem::is_directory(dir_path) || boost::filesystem::is_empty(dir_path))
		throw std::string("Non existing or empty dataset directory!");
	imgDirectory = dir_path;
	vector<string> names = listFilesInDirectory(dir_path, fileExtension);
	if (names.size()==0)
		throw std::string("No image found in directory!");
	copy(names.begin(), names.end(),back_inserter(ImgNames));
}

unsigned long Vocabulary::extractFeatures_TS(const string &DetectorType, const string &DescriptorType) throw(std::runtime_error){
	FeatureDetectorType = DetectorType;
	generateFolders(kp_folder, descr_folder, outdata);
	unsigned long n_of_features = 0;
	size_t img_counter=0;            // iteratively incremented param used for keypoints output file generation
	string path_temp, key_points_directory(outputFolder+kp_folder+"/"), key_points_extension(".keypoint");
	vector<cv::KeyPoint> keypoints;
	TotalImagesNumber=0;
	vector<string>::iterator names_iter = ImgNames.begin(), names_iter_fal = ImgNames.end();
	for(;names_iter!=names_iter_fal; ++names_iter) {
		path_temp = *names_iter;
		Ptr<FeatureDetector> featureDetector;
		if(DetectorType=="FAST" && (DescriptorType=="BRIEF"||DescriptorType=="BRIEF64"))
			featureDetector = new cv::FastFeatureDetector(50);
		if(DetectorType!="FAST")
			featureDetector = FeatureDetector::create(DetectorType);
		bool grayscale = true;
		cv::Mat img_gray = grabImage(path_temp, grayscale);
		if(img_gray.empty())
			throw runtime_error("Error in loading image" +path_temp);
		keypoints = getImgFeatures(img_gray, featureDetector);
		n_of_features+=keypoints.size();
		img_gray.release();
		if (keypoints.size()==0){
			ImgNames.erase(names_iter,names_iter+1);
			names_iter = ImgNames.begin()+img_counter;
			names_iter_fal = ImgNames.end();
			continue;
		}
		KeyPointsByImage.push_back(keypoints);
		cout<<"features found in the image:"<<keypoints.size()<<endl;
		++TotalImagesNumber;
		path_temp=generatePath(key_points_directory, img_counter,key_points_extension);
		ofstream img_keypoints;
		img_keypoints.open(path_temp.c_str());
		if (!img_keypoints) {
			cerr <<"error: unable to open output file: "<<path_temp<<endl;
		}
		vector<cv::KeyPoint>::iterator itr = keypoints.begin(), itr_fal = keypoints.end();
		for(;itr!=itr_fal;++itr){
			img_keypoints<<itr->pt.x<<" "<<itr->pt.y<<endl;
		}
		img_keypoints.close();
		keypoints.clear();
		++img_counter;
	}
	return n_of_features;
}

unsigned long Vocabulary::describeFeatures_TS(const string &DescriptorType ) throw(std::runtime_error){
	DescriptorExtractorType = DescriptorType;
	string path_temp;
	size_t img_count = 0;
	string d_base(outputFolder+descr_folder+"/"), d_extension(".descr");
	vector<string>::const_iterator names_iter = ImgNames.begin(), names_iter_fal = ImgNames.end();
	cv::Mat img_gray, descriptors;
	for (;names_iter!=names_iter_fal;++names_iter){
		bool grayscale (DescriptorType=="OpponentSURF"||DescriptorType=="OpponentSIFT" ||DescriptorType=="OpponentBRIEF");
		img_gray = grabImage(*names_iter, grayscale);
		if(DescriptorType=="SURF128"){
			descriptors = describeImgSurfFeatures(img_gray, KeyPointsByImage.front(), 4,2,1);
		}
		if(DescriptorType=="SURF"){
			descriptors = describeImgSurfFeatures(img_gray, KeyPointsByImage.front(), 4,2,0);
		}
		if(DescriptorType=="BRIEF64"){
			descriptors = describeImgBriefFeatures(img_gray, KeyPointsByImage.front(), 64);
		}
		if(DescriptorType!="BRIEF64" && DescriptorType!="SURF128"&& DescriptorType!="SURF"){
			cv::Ptr<cv::DescriptorExtractor> descriptorExtractor = cv::DescriptorExtractor::create(DescriptorType);
			descriptors = describeImgFeatures (img_gray, KeyPointsByImage.front(), descriptorExtractor);
			descriptorExtractor.~Ptr();
		}
		img_gray.~Mat();
		AllDescriptorsByImage.push_back(descriptors);
		TotalDescriptorNumber+=descriptors.rows;
		string path_temp = generatePath(d_base,img_count,d_extension);
		MatToFile(path_temp, descriptors);
		cout<<"descriptors type is "<<descriptors.type()<<endl;
		descriptors.~Mat();
		KeyPointsByImage.pop_front();				// <<----deleting key points
		++img_count;
	}
	return TotalDescriptorNumber;
}


unsigned long Vocabulary::clusterDescriptors(const string &method,const int &number_cluster) throw(std::runtime_error) {
	if(number_cluster==0 || number_cluster<0) {
		throw runtime_error("Invalid argument for clustering");
	}
	unsigned long count=0;
	cout<<"Trying to calculate great matrix"<<endl;
	while (!AllDescriptorsByImage.empty()){
		if (count==0) {
			GreatMatrix= AllDescriptorsByImage.front();
			AllDescriptorsByImage.pop_front();
		}
		else {
			GreatMatrix.push_back(AllDescriptorsByImage.front());
			AllDescriptorsByImage.pop_front();
		}
		++count;
	}
	cout<<"GreatMatrix calculated from "<<ImgNames.size()<<" images "<<endl;
	cout<<"Total n. of descriptors: "<<GreatMatrix.rows<<endl;
	dataDimension = GreatMatrix.cols;
	if(method=="kMeansClustering"){
		cv::TermCriteria term_crit = cv::TermCriteria(0,50,0.01);
		int attempts = 10;
		int flags = 2;
		GreatMatrix.convertTo(GreatMatrix,CV_32FC1);
		cv::kmeans (GreatMatrix, number_cluster, PointToCluster, term_crit, attempts, flags, cv::_OutputArray(ClusterCenters));
		string path= outputFolder+ outdata+ "PointToCluster.out";
		MatToFile(path, PointToCluster);
	}
	if(method=="hierarchicalClustering"){
		Mat clCenters(number_cluster,GreatMatrix.cols,CV_32FC1);
		cvflann::KMeansIndexParams parameters(32,11,cvflann::CENTERS_RANDOM,0.2);
		int total_words=0;
		cout<<GreatMatrix.cols*GreatMatrix.rows<<endl;
		if(DescriptorExtractorType=="BRIEF" || DescriptorExtractorType=="BRIEF64"){
			total_words = flann::hierarchicalClustering<cv::flann::L2<unsigned char> > (GreatMatrix, clCenters, parameters);
		}
		else{
			total_words = flann::hierarchicalClustering<cv::flann::L2<float> >(GreatMatrix, clCenters, parameters);
		}
		ClusterCenters = clCenters;
		if(total_words<number_cluster) {
			cout<<"Trying to truncate the vocabulary matrix"<<endl;
			cout<<"Number of clusters"<<number_cluster<<endl;
			cout<<"Total words"<<total_words<<endl;
			ClusterCenters = clCenters.rowRange(0,total_words-1);
			cout<<"Updated total words"<<total_words-1<<endl;
		}
		clCenters.~Mat();
	}
	GreatMatrix.~Mat();
	/*string vocabSavePath(outputFolder+outdata+"vocabulary.out");

	ofstream print_to_file;
	print_to_file.open(vocabSavePath.c_str());
	if (!print_to_file) {
		throw runtime_error("Unable to open output file: "+vocabSavePath);
	}
	print_to_file<<"DETECTOR_TYPE: "<<FeatureDetectorType<<endl;
	print_to_file<<"DESCRIPTOR_TYPE: "<<DescriptorExtractorType<<endl;
	print_to_file<<"WORDS: "<<ClusterCenters.rows<<endl;
	print_to_file<<"DESCRIPTOR_DIMENSION: "<<dataDimension<<endl;
	print_to_file<<"WORD:0 "<<endl;
	int new_line =0;
	cv::MatConstIterator_<float> iter_mat = ClusterCenters.begin<float>(),iter_mat_fal=ClusterCenters.end<float>();
	for (;iter_mat!=iter_mat_fal;++iter_mat){
		if (new_line!=0 && new_line%ClusterCenters.cols == 0) {
			print_to_file<<"\n"<<flush;
			print_to_file<<"WORD:"<<new_line/ClusterCenters.cols<<endl;
		}
		print_to_file<<*iter_mat<<" "<<flush;
		++new_line;
	}
	print_to_file.close();*/
	saveVocab(ClusterCenters, outputFolder+outdata+"vocabulary.out");
	return ClusterCenters.rows;
}

void Vocabulary::buildAllIndeces (const string& Index_path, const string& InvIndex_path) throw (runtime_error){
	cout<<"Generating flann index"<<endl;
	cvflann::AutotunedIndexParams flannIndexParams(0.8,0.01,0, 0.1);
	cv::flann::GenericIndex<cv::flann::L2<float> > flannIndex (ClusterCenters, flannIndexParams); //(cvflann::IndexParams &)flannIndexParams);
	cvflann::SearchParams knnSearchParams(32); //The number of times the tree(s) in the index should be recursively traversed.
	vector<string> names = listFilesInDirectory(outputFolder+descr_folder,".descr");
	sort(names.begin(),names.end());
	size_t counter_d = 0;
	vector<string>::const_iterator dscr_iter = names.begin(), dscr_iter_fal = names.begin()+ImgNames.size();
	for(;dscr_iter!=dscr_iter_fal;++dscr_iter){
		ifstream readingImgDescriptors(dscr_iter->c_str());
		cout<<*dscr_iter<<endl;
		if(!readingImgDescriptors)
			throw runtime_error("No file for descriptor found");
		cout<<"Reading descriptor "<<*dscr_iter<<endl;
		int descriptors_per_image=0, dim =0;
		unsigned temp_coord = 0;
		float temp_coord_float = 0;
		string tt, trash;
		stringstream buffer;
		getline (readingImgDescriptors, tt);
		buffer<<tt<<endl;
		buffer>>trash>>descriptors_per_image;
		buffer.clear();
		getline (readingImgDescriptors, tt);
		buffer<<tt<<endl;
		buffer>>trash>>dim;
		buffer.clear();
		cout<<descriptors_per_image<<" "<<dim<<endl;
		cv::Mat curImageDescriptors(descriptors_per_image,dim, CV_32FC1);
		cv::MatIterator_<float> it_mat = curImageDescriptors.begin<float>();
		cout<<DescriptorExtractorType<<endl;
		if (DescriptorExtractorType=="BRIEF"||DescriptorExtractorType=="BRIEF64"){
			while (getline(readingImgDescriptors, tt)){
				buffer<<tt<<endl;
				while(buffer>>temp_coord){
					(*it_mat) = temp_coord;
					++it_mat;
				}
				buffer.clear();
			}
		}
		else {
			while (getline(readingImgDescriptors, tt)){
				buffer<<tt<<endl;
				while(buffer>>temp_coord_float){
					(*it_mat) = temp_coord_float;
					++it_mat;
				}
				buffer.clear();
			}
		}
		cv::Mat flann_labels_inner(curImageDescriptors.rows, 1, CV_32SC1);		// <-- int
		cv::Mat flann_dist_inner (curImageDescriptors.rows, 1, CV_32FC1);		// <-- float
		flannIndex.knnSearch(curImageDescriptors, flann_labels_inner, flann_dist_inner, 1, knnSearchParams);
		curImageDescriptors.~Mat();
		cv::MatIterator_<int> lab_iter = flann_labels_inner.begin<int>(),
				lab_iter_fal = flann_labels_inner.end<int>();
		map<int, int> freq_map;
		pair<int, vector<int> > sowInvertIndex_pair;
		vector<int> initializer;
		pair<map<int, vector<int> >::iterator, bool> sowInvertIndex_write;
		for(;lab_iter!=lab_iter_fal;++lab_iter){
			//update std::map<int, int> vw_occurrence_n;
			++vw_occurrence_n[*lab_iter];
			++freq_map[*lab_iter];
			bool seen_before =false;
			initializer.push_back(counter_d);
			sowInvertIndex_pair=make_pair(*lab_iter, initializer);
			sowInvertIndex_write = sowInvertIndex.insert(sowInvertIndex_pair);
			if(!sowInvertIndex_write.second){
				vector<int>::iterator   inner_runner=sowInvertIndex_write.first->second.begin(),
						inner_runner_fal=sowInvertIndex_write.first->second.end();
				for(;inner_runner!=inner_runner_fal;++inner_runner){
					size_t lulz = *inner_runner;
					if (counter_d==lulz) {
						seen_before=true;
						break;
					}
				}
				if (seen_before==false) sowInvertIndex_write.first->second.push_back(counter_d);
			}
			initializer.clear();
		}
		sowIndex[counter_d]=freq_map;
		++counter_d;
	}
	names.clear();
	String temp = outputFolder+ outdata+ Index_path;
	ofstream print_index(temp.c_str());
	if(!print_index){
		throw runtime_error("Cannot open file "+outputFolder+outdata+Index_path+ "for writing");
	}
	map<int, map<int,int> >::iterator   iter_crazy=sowIndex.begin(),
			iter_crazt_fal=sowIndex.end();
	for(;iter_crazy!=iter_crazt_fal;++iter_crazy){
		print_index<<"image "<<iter_crazy->first<<endl;
		map<int, int >::iterator     iter_inner=iter_crazy->second.begin(),
				iter_inner_fal=iter_crazy->second.end();
		for(;iter_inner!=iter_inner_fal;++iter_inner){
			print_index<<iter_inner->first<<" "<<iter_inner->second<<" "<<flush;
		}
		print_index<<"\n"<<flush;
	}
	print_index.close();
	cout<<"Printing "<<outputFolder+outdata+Index_path<<" done successfully"<<endl;
	map<int, int>::iterator freaq_itr = vw_occurrence_n.begin(),
			freaq_itr_fal = vw_occurrence_n.end();
	temp = outputFolder + outdata+ "vw_occurrence_n_general.out";
	ofstream print_freaq(temp.c_str());

	for(; freaq_itr!=freaq_itr_fal;++freaq_itr){
		print_freaq<<freaq_itr->second<<endl;
	}
	print_freaq.close();
	cout<<"Printing "<<outputFolder+ outdata+ "/vw_occurrence_n_general.out done successfully"<<endl;
	temp = outputFolder+outdata+InvIndex_path;
	ofstream inv_ind_print(temp.c_str());
	if (!inv_ind_print){
		throw runtime_error("inv_ind_print: cannot open file for writing");
	}
	map<int, vector<int> >::iterator   iter_inv_index = sowInvertIndex.begin(),
			iter_inv_index_fal=sowInvertIndex.end();
	for(; iter_inv_index!=iter_inv_index_fal;++iter_inv_index){
		vector<int>::iterator inner_runner = iter_inv_index->second.begin(),
				inner_runner_fal=iter_inv_index->second.end();
		for(;inner_runner!=inner_runner_fal;++inner_runner){
			inv_ind_print<<*inner_runner<<" "<<flush;
		}
		inv_ind_print<<"\n"<<flush;
	}
	inv_ind_print.close();
	cout<<"BuildInverseIndex: printing process finished"<<endl;
	return;
}

void Vocabulary::truncateVocabulary(const float upperLimit, const float lowerLimit) throw (runtime_error){
	cout<<"vocabulary truncation started"<<endl;
	if (vw_occurrence_n.size()==0)
		throw runtime_error("Empty vw_occurrence_n map");
	if (dataDimension==0)
		throw runtime_error ("No dataDimension info");
	deque<pair<int,int> > vw_times_seen_vect;
	map<int,int>::const_iterator map_it = vw_occurrence_n.begin(), map_it_fal = vw_occurrence_n.end();
	for(;map_it!=map_it_fal;++map_it){
		vw_times_seen_vect.push_back(*map_it);
	}
	sort (vw_times_seen_vect.begin(), vw_times_seen_vect.end(), comparator);
	int scip_first_n = upperLimit*vw_times_seen_vect.size();
	int scip_last_m = lowerLimit*vw_times_seen_vect.size();
	for(int i=0; i!=scip_first_n;++i){
		vw_times_seen_vect.pop_front();
	}
	for(int j=0; j!=scip_last_m;++j){
		vw_times_seen_vect.pop_back();
	}
	cout<<"Copying data from old vocabulary .... "<<endl;
	cv::Mat newVocab (vw_times_seen_vect.size(),dataDimension, CV_32FC1);
	deque<pair<int,int> >::const_iterator deq_trunc = vw_times_seen_vect.begin(),
			deq_trunc_fal = vw_times_seen_vect.end();
	cv::MatIterator_<float> new_it = newVocab.begin<float>(), new_it_fal = newVocab.end<float>();
	cv::MatIterator_<float> old_begin = ClusterCenters.begin<float>(), old_temp = ClusterCenters.begin<float>();
	for (; deq_trunc!=deq_trunc_fal; ++deq_trunc){
		old_temp = old_begin+ (deq_trunc->first)*dataDimension;
		for(unsigned i=0; i!=dataDimension;++i){
			*new_it = *old_temp;
			++new_it;
			++old_temp;
		}
	}
	/*String vocabSavePath(outputFolder+outdata+"vocabulary.out");
	ofstream print_to_file;
	print_to_file.open(vocabSavePath.c_str());
	if (!print_to_file) {
		cerr <<"error: unable to open output file: "<<vocabSavePath<<endl;
	}
	print_to_file<<"DETECTOR_TYPE: "<<FeatureDetectorType<<endl;
	print_to_file<<"DESCRIPTOR_TYPE: "<<DescriptorExtractorType<<endl;
	print_to_file<<"WORDS: "<<newVocab.rows<<endl;
	print_to_file<<"DESCRIPTOR_DIMENSION: "<<dataDimension<<endl;
	print_to_file<<"WORD:0 "<<endl;
	int new_line =0;
	cv::MatConstIterator_<float> iter_mat = newVocab.begin<float>(),iter_mat_fal=newVocab.end<float>();
	for (;iter_mat!=iter_mat_fal;++iter_mat){
		if (new_line!=0 && new_line%newVocab.cols == 0) {
			print_to_file<<"\n"<<flush;
			print_to_file<<"WORD:"<<new_line/newVocab.cols<<endl;
		}
		print_to_file<<*iter_mat<<" "<<flush;
		++new_line;
	}
	print_to_file.close();*/
	saveVocab(newVocab, outputFolder+outdata+"vocabulary.out");
	newVocab.~Mat();
}



void Vocabulary::readFabMapDict(const string &tDictionary){	 
	try{
		ifstream rdInvIndex(tDictionary.c_str());
		if(!rdInvIndex) throw runtime_error("no file");
		string dataLine;
		int total_n_words=0;
		getline(rdInvIndex, dataLine);
		stringstream convert;
		for(string::size_type i=0; i!=dataLine.size();++i){
			if(isdigit(dataLine[i]))  convert<<dataLine[i];
		}
		rdInvIndex.close();

		convert>>total_n_words;
		dataDimension = 128; //<--- !!! probably is to be read also
		DescriptorExtractorType="SURF";

		ClusterCenters = cv::Mat (total_n_words,dataDimension,CV_32FC1);
		cv::MatIterator_<float>   iter = ClusterCenters.begin<float>(),
				iter_end=ClusterCenters.end<float>();

		float vocValue=0;
		ifstream reader(tDictionary.c_str());

		unsigned line_counter=1;
		while(getline(reader,dataLine)){

			if(dataLine.size()>20){

				stringstream convert;
				convert<<dataLine<<flush;

				while(convert>>vocValue) {
					*iter=vocValue;
					++iter;
				}
				convert.clear();
			}

			++line_counter;
		}

		rdInvIndex.close();
		cout<<"Dictionary read with "<<ClusterCenters.rows<<" words"<<endl;
	}
	catch(runtime_error& err) {
		cout<<" !error: "<<err.what()<<" "<<tDictionary<<" found"<<endl;
		return;
	}
}

unsigned long Vocabulary::readDict(const string &tDictionary) throw (runtime_error){
	ifstream reader(tDictionary.c_str());
	if(!reader){
		throw runtime_error("Dictionary file not found");
	}
	string dataLine, temp;
	stringstream convert;
	int total_n_words;
	getline(reader, dataLine);
	convert<<dataLine;
	convert>>temp>>FeatureDetectorType;
	convert.clear();
	getline(reader, dataLine);
	convert<<dataLine;
	convert>>temp>>DescriptorExtractorType;
	convert.clear();
	getline(reader, dataLine);
	for(string::size_type i=0; i!=dataLine.size();++i){
		if(isdigit(dataLine[i]))  convert<<dataLine[i];
	}
	convert>>total_n_words;
	convert.clear();
	getline(reader, dataLine);
	for(string::size_type i=0; i!=dataLine.size();++i){
		if(isdigit(dataLine[i]))  convert<<dataLine[i];
	}
	convert>>dataDimension;
	convert.clear();
	ClusterCenters = cv::Mat (total_n_words,dataDimension,CV_32FC1);
	cv::MatIterator_<float>   iter = ClusterCenters.begin<float>(),
			iter_end=ClusterCenters.end<float>();
	float vocValue=0;
	unsigned line_counter=1;
	while(getline(reader,dataLine)){
		if(dataLine.size()>20){
			stringstream convert;
			convert<<dataLine<<flush;
			while(convert>>vocValue) {
				*iter=vocValue;
				++iter;
			}
			convert.clear();
		}
		++line_counter;
	}
	reader.close();
	return ClusterCenters.rows;
}

vector<int> Vocabulary::getSowBowRepr_img( const cv::Mat &current_picture,const int &method,						// why do i need a method here??
		cv::flann::GenericIndex<cv::flann::L2<float> >  &flannIndObj,
		vector<cv::KeyPoint> &curImgFeatures,
		cv::Mat &curImgDescriptors,
		cv::Mat &flann_labels,
		cv::Mat &flann_dist) const{
	try{
		vector<int> bow_repres(ClusterCenters.rows,-1);
		int flann_knn = 1;
		cvflann::SearchParams knnSearchParams(32);			  	// should be user-defined (not necessarily)
		cv::Ptr<cv::FeatureDetector> featureDetector = cv::FeatureDetector::create(FeatureDetectorType);
		bool grayscale = true;									// features are always detected i b-w image
		cv::Mat img_gray;
		cv::cvtColor (current_picture, img_gray, CV_RGBA2GRAY);
		curImgFeatures = getImgFeatures(img_gray, featureDetector);
		if (curImgFeatures.empty()){
			cout<<"!!!! THIS IMAGE HAS NO FEATURES !!!!"<<endl;
			return bow_repres;
		}
		cv::Ptr<cv::DescriptorExtractor> extractor = cv::DescriptorExtractor::create(DescriptorExtractorType);
		grayscale = (DescriptorExtractorType=="SURF"||DescriptorExtractorType=="SURF128" || DescriptorExtractorType=="SIFT" ||  DescriptorExtractorType=="BRIEF"||  DescriptorExtractorType=="BRIEF64");
		//cout<<"describeImgFeatures started"<<endl;
		if(!grayscale){
			cv::cvtColor(current_picture, img_gray, CV_RGBA2BGR,3 );		//CV_RGBA2BGRA
		}
		if( (DescriptorExtractorType=="SURF" && dataDimension==128) || DescriptorExtractorType=="SURF128"){
			curImgDescriptors = describeImgSurfFeatures(img_gray, curImgFeatures, 4,2,1);
			//cout<<"surf128: "<<curImgDescriptors.rows<<" "<<curImgDescriptors.cols<<endl;
		}
		else if( DescriptorExtractorType=="SURF" ){
			curImgDescriptors = describeImgSurfFeatures(img_gray, curImgFeatures, 4,2,0);
			//cout<<"surf: "<<curImgDescriptors.rows<<" "<<curImgDescriptors.cols<<endl;
		}
		else if( (DescriptorExtractorType=="BRIEF" && dataDimension==64) || DescriptorExtractorType=="BRIEF64"){
			curImgDescriptors=describeImgBriefFeatures(img_gray, curImgFeatures, 64);
			//cout<<"brief64: "<<curImgDescriptors.rows<<" "<<curImgDescriptors.cols<<endl;
		}
		else {
			curImgDescriptors = describeImgFeatures (img_gray, curImgFeatures, extractor);
			//cout<<DescriptorExtractorType<<" "<<curImgDescriptors.rows<<" "<<curImgDescriptors.cols<<endl;
		}
		img_gray.~Mat();
		cv::Mat flann_labels_inner (curImgDescriptors.rows, flann_knn, CV_32SC1);		// <-- int
		cv::Mat flann_dist_inner (curImgDescriptors.rows, 1, CV_32FC1);					// <-- float
		curImgDescriptors.convertTo(curImgDescriptors,CV_32FC1);
		bow_repres = getFlannSowBowVector (flannIndObj, curImgDescriptors, flann_labels_inner, flann_dist_inner, flann_knn, knnSearchParams, 1);
		flann_labels = flann_labels_inner.clone();
		flann_labels_inner.~Mat();
		flann_dist = flann_dist_inner.clone();
		flann_dist_inner.~Mat();
		return bow_repres;
	}
	catch (runtime_error& err){
		vector<int> err_bow;
		cout<<err.what()<<endl;
		return err_bow;
	}
}

void Vocabulary::generateFolders(const string &kp_folder, const string &desc_folder, const string &outdata) const{
	if (!boost::filesystem::exists(outputFolder))
		boost::filesystem::create_directory(outputFolder);
	if (!boost::filesystem::exists(outputFolder+kp_folder))
		boost::filesystem::create_directory(outputFolder+kp_folder);
	if (!boost::filesystem::exists(outputFolder+desc_folder))
		boost::filesystem::create_directory(outputFolder+desc_folder);
	if (!boost::filesystem::exists(outputFolder+outdata))
		boost::filesystem::create_directory(outputFolder+outdata);
}


inline string Vocabulary::generatePath(const string &inp_base, const size_t &number, const string &inp_extension) const {
	stringstream converter;
	string address, extCounter;
	int temp_number = number, digits=0;
	if (number==0) ++digits;
	while (temp_number!=0){
		temp_number/=10;
		++digits;
	}
	for (int i=0; i!=7-digits;++i){
		extCounter=extCounter+"0";
	}
	converter<<inp_base<<extCounter<<number<<inp_extension<<flush;
	converter>>address;
	return address;
}

cv::Mat Vocabulary::describeImgFeatures (const cv::Mat &img_gray, vector<cv::KeyPoint> &key_points,	const cv::Ptr<cv::DescriptorExtractor> &descriptorExtractor) const {
	cv::Mat descriptors;
	descriptorExtractor->compute(img_gray,key_points,descriptors);
	//cout<<"describeImgFeatures: computed "<<descriptors.rows<<" descriptors"<<endl;
	return descriptors;
}

vector<cv::KeyPoint> Vocabulary::getImgFeatures(const cv::Mat &img_gray, const cv::Ptr<cv::FeatureDetector> &detector) const {
	//cout<<"getImgFeatures started"<<endl;
	vector<cv::KeyPoint> key_points;
	detector->detect(img_gray,key_points);
	//cout<<"getImgFeatures: key_points extracted, found "<<key_points.size()<<" key points "<<endl;
	return key_points;
}

cv::Mat Vocabulary::grabImage(const string &path, const int &grayscale) const{
	cv::Mat img = cv::imread(path.c_str());
	if(!img.data)
		return img;
	cv::Mat img_gray(img);
	if (grayscale==0)
		cv::cvtColor (img, img_gray, CV_RGBA2GRAY );
	if (grayscale==1)
		cv::cvtColor (img, img_gray, CV_RGBA2BGR,3 );		//CV_RGBA2BGRA
	img.~Mat();
	return img_gray;
}

vector<string> Vocabulary::listFilesInDirectory (const std::string &dir_path, const std::string &ext) const {
	vector<string> filesInDirectory;
	boost::filesystem::directory_iterator end_itr;
	for (boost::filesystem::directory_iterator itr(dir_path);itr!=end_itr;++itr) {
		if(boost::filesystem::is_regular_file(*itr) && itr->path().extension() == ext){
			std::string tmp=itr->path().filename().string();
			string absPath = dir_path+"/"+tmp;
			filesInDirectory.push_back(absPath);
		}
	}
	sort(filesInDirectory.begin(),filesInDirectory.end());
	return filesInDirectory;
}


void Vocabulary::MatToFile(const string &doc_path, cv::Mat matrix) const{

	matrix.convertTo(matrix,CV_32FC1);

	//cout<<"MatToFile printing started"<<endl;
	ofstream print_to_file;
	print_to_file.open(doc_path.c_str());
	if (!print_to_file) {
		cerr <<"error: unable to open output file: "<<doc_path<<endl;
	}
	print_to_file<<"N_ROWS: "<<matrix.rows<<endl;
	print_to_file<<"N_COLS: "<<matrix.cols<<endl;

	int new_line =1;
	cv::MatConstIterator_<float> iter_mat = matrix.begin<float>(),iter_mat_fal=matrix.end<float>();

	for (;iter_mat!=iter_mat_fal;++iter_mat){
		print_to_file<<*iter_mat<<" "<<flush;
		if (new_line%matrix.cols == 0) {
			print_to_file<<"\n"<<flush;
		}
		++new_line;
	}
	matrix.release();
	print_to_file.close();

}

vector<int> Vocabulary::getFlannSowBowVector(cv::flann::GenericIndex<cv::flann::L2<float> >  &flannIndex,
		Mat &descr_query,
		Mat& indices,
		Mat &dist,
		int knn,
		cvflann::SearchParams params,
		int sow_bow) const {
	vector<int> sow_vector (ClusterCenters.rows,0);
	flannIndex.knnSearch(descr_query, indices, dist, knn, params);
	cv::MatConstIterator_<int> labels_iter = indices.begin<int>(), labels_iter_fal = indices.end<int>();
	vector<int>::iterator itr_vect = sow_vector.begin(), itr_vect_temp = sow_vector.end();
	for (; labels_iter!=labels_iter_fal;++labels_iter){
		itr_vect_temp = itr_vect+ (*labels_iter);
		if (itr_vect_temp < sow_vector.end()){
			if (sow_bow == 0) *itr_vect_temp = 1;
			if (sow_bow == 1) (*itr_vect_temp)++;
		}
		else { throw runtime_error("itr_vect_temp is greater then the vector");}
	}
	return sow_vector;
}

cv::Mat Vocabulary::describeImgSurfFeatures(const cv::Mat &img_gray, vector<cv::KeyPoint> &key_points,	const int nOctaves,	const int nOctaveLayers, const bool extended) const {
	cv::Mat descriptors;
	cv:SURF surfer(500,nOctaves, nOctaveLayers,extended,false);
	surfer.compute(img_gray,key_points,descriptors);
	cout<<"describeImgSurfFeatures: computed "<<descriptors.rows<<" descriptors"<<endl;
	return descriptors;
}

cv::Mat Vocabulary::describeImgBriefFeatures(const cv::Mat &img_gray, vector<cv::KeyPoint> &key_points, const int bits_n) const {
	cv::Mat descriptors;
	cv::BriefDescriptorExtractor briefer(bits_n);
	briefer.compute(img_gray,key_points,descriptors);
	cout<<"describeImgSurfFeatures: computed "<<descriptors.rows<<" descriptors"<<endl;
	return descriptors;
}

bool comparator (const pair<int,int> &l, const pair<int,int> &r){
	return l.second > r.second;
}

void Vocabulary::saveVocab(cv::Mat newVocab, string dir){
	if(newVocab.empty())
		newVocab = ClusterCenters;
	String vocabSavePath(dir);
	ofstream print_to_file;
	print_to_file.open(vocabSavePath.c_str());
	if (!print_to_file) {
		cerr <<"error: unable to open output file: "<<vocabSavePath<<endl;
	}
	print_to_file<<"DETECTOR_TYPE: "<<FeatureDetectorType<<endl;
	print_to_file<<"DESCRIPTOR_TYPE: "<<DescriptorExtractorType<<endl;
	print_to_file<<"WORDS: "<<newVocab.rows<<endl;
	print_to_file<<"DESCRIPTOR_DIMENSION: "<<dataDimension<<endl;
	print_to_file<<"WORD:0 "<<endl;
	int new_line =0;
	cv::MatConstIterator_<float> iter_mat = newVocab.begin<float>(),iter_mat_fal=newVocab.end<float>();
	for (;iter_mat!=iter_mat_fal;++iter_mat){
		if (new_line!=0 && new_line%newVocab.cols == 0) {
			print_to_file<<"\n"<<flush;
			print_to_file<<"WORD:"<<new_line/newVocab.cols<<endl;
		}
		print_to_file<<*iter_mat<<" "<<flush;
		++new_line;
	}
	print_to_file.close();
}
