#include "tracker_summer.h"

#include "ar_utils/pose_converter/pose_converter.h"
#include "ar_utils/data_io/data_loader.h"

#include <memory>
#include <cstdlib>

using namespace ar3dv;

cv::Matx33f PolarDecomposition(const cv::Matx33f &M)
{
	cv::Matx33f R = M;
	for (int i = 0; i < 10; ++i)
	{ // Iterate for refinement
		cv::Matx33f R_next = (R + R.t().inv()) * 0.5;
		if (cv::norm(R_next - R) < 1e-6)
		{
			break;
		}
		R = R_next;
	}
	return R;
}

void NormalizePoseByPD(cv::Matx44f &pose)
{
	cv::Matx33f rotation;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			rotation(i, j) = pose(i, j);
		}
	}

	cv::Matx33f orthogonalizedRotation = PolarDecomposition(rotation);

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			pose(i, j) = orthogonalizedRotation(i, j);
		}
	}
	return;
}

void NormalizePose(cv::Matx44f &pose)
{
	VLOG(0) << "pose before NormalizePose \n"
			<< pose;
	cv::Matx33f rotation = cv::Matx33f::eye();
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			rotation(i, j) = pose(i, j);
		}
	}

	cv::Mat w, u, vt;
	cv::SVD::compute(cv::Mat(rotation), w, u, vt);
	cv::Mat orthogonalizedRotation = u * vt;

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			pose(i, j) = orthogonalizedRotation.at<float>(i, j);
		}
	}

	// Optionally you can enforce a valid rotation matrix by checking the determinant
	cv::Matx33f finalRotation;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			finalRotation(i, j) = pose(i, j);
		}
	}
	float det = cv::determinant(cv::Mat(finalRotation));
	if (fabs(det - 1.0f) > 1e-3)
	{
		// Log or handle error: the matrix is not a valid rotation matrix
		std::cerr << "Warning: Resulting rotation matrix is not orthogonal." << std::endl;
	}
	return;
}

SummerTracker::SummerTracker() : TrackerBase()
{
	VLOG(0) << "Summer starts!";
}

SummerTracker::~SummerTracker()
{
	VLOG(0) << "Summer has ended!";
}

void SummerTracker::StartTracking()
{
	StdData *currData = m_stdDatas[0];
	if (m_startTracking)
	{
		VLOG(0) << "Start Tracking";
		m_objects[0]->setPose(currData->gt);
		tracker_->ToggleTracking(currData->frame, 0, false);
		m_startTracking = false;
	}
}

void SummerTracker::DoNotEstimate()
{
	StdData *currData = m_stdDatas[0];
	NormalizePose(currData->gt);
	StartTracking();
	m_objects[0]->setPose(currData->gt);
	tracker_->UpdateHist(currData->frame);
	cv::Mat result = renderer_->DrawResultOverlay(std::vector<summer::Model *>(m_objects.begin(), m_objects.end()), currData->frame);
	SetTrackingResult(ResultType::kResOverlay, result);
	SetTrackingResult(ResultType::kResPose, m_objects[0]->getPose());
}

void SummerTracker::GenRefers()
{
	PrintInfo();
	StdData *currData = m_stdDatas[0];

	StartTracking();
	tracker_->EstimatePoses(currData->frame, 2);
	m_objects[0]->setPose(currData->gt);
	tracker_->UpdateHist(currData->frame);
	cv::Matx44f currPose = m_objects[0]->getPose();

	cv::Mat prob = tracker_->prob_map(currData->frame, 0);
	cv::Mat probDeepCopy = prob.clone();
	cv::Mat frameDeepCopy = currData->frame.clone();

	DataGroup dataGroup;
	dataGroup.valid = false;
	dataGroup = ComDataGroup(frameDeepCopy, probDeepCopy, true);
	SetTrackingResult(ResultType::kResDataGroup, dataGroup);

	cv::Mat result = renderer_->DrawResultOverlay(std::vector<summer::Model *>(m_objects.begin(), m_objects.end()), currData->frame, true);
	cv::imshow("result", result);
}

void SummerTracker::Estimate()
{
	PrintInfo();
	StdData *currData = m_stdDatas[0];

	StartTracking();
	tracker_->EstimatePoses(currData->frame, 2);
	tracker_->UpdateHist(currData->frame);

	cv::Mat prob;

	cv::Matx44f currPose = m_objects[0]->getPose();

	bool success = IsSuccess(currPose);

	prob = tracker_->prob_map(currData->frame, 0);
	cv::Mat probDeepCopy = prob.clone();
	SetTrackingResult(ResultType::kResValid, 1);
	cv::Mat frameDeepCopy = currData->frame.clone();

	DataGroup dataGroup;
	dataGroup.valid = false;
	dataGroup = ComDataGroup(frameDeepCopy, probDeepCopy);
	if (probDeepCopy.type() == CV_8UC1 && !probDeepCopy.empty())
	{
		double residualSum = 0.0;
		double noiseVarianceSum = 0.0;
		double separationSum = 0.0;
		int sampleCount = 0;
		for (int row = 0; row < probDeepCopy.rows; ++row)
		{
			const uchar *values = probDeepCopy.ptr<uchar>(row);
			for (int col = 0; col < probDeepCopy.cols; ++col)
			{
				const double foreground = values[col] / 255.0;
				noiseVarianceSum += foreground * (1.0 - foreground);
				residualSum += -std::log(std::max(1e-7, std::max(foreground, 1.0 - foreground)));
				separationSum += std::abs(2.0 * foreground - 1.0);
				++sampleCount;
			}
		}
		dataGroup.contour_residual = residualSum / sampleCount;
		dataGroup.histogram_separation = separationSum / sampleCount;
		dataGroup.contour_noise_variance = noiseVarianceSum / sampleCount;
		dataGroup.contour_samples = sampleCount;
	}
	dataGroup.pose_hessian = tracker_->pose_hessian();
	dataGroup.pose_hessian_valid = tracker_->pose_hessian_valid();
	const cv::Vec3f modelExtent =
		(m_objects[0]->getRTF() - m_objects[0]->getLBN()) *
		m_objects[0]->getScaling();
	dataGroup.object_characteristic_length = cv::norm(modelExtent);
	if (dataGroup.pose_hessian_valid)
	{
		cv::Mat eigenvalues;
		if (cv::eigen(cv::Mat(dataGroup.pose_hessian), eigenvalues))
		{
			dataGroup.hessian_max_eigenvalue = eigenvalues.at<float>(0);
			dataGroup.hessian_min_eigenvalue = eigenvalues.at<float>(
				eigenvalues.rows - 1);
			const float minimum = std::max(
				dataGroup.hessian_min_eigenvalue, 1e-12f);
			dataGroup.hessian_condition =
				dataGroup.hessian_max_eigenvalue / minimum;
			cv::Matx66f regularized = dataGroup.pose_hessian;
			const float damping = std::max(
				1e-9f, dataGroup.hessian_max_eigenvalue * 1e-6f);
			for (int axis = 0; axis < 6; ++axis)
				regularized(axis, axis) += damping;
			const cv::Matx66f covariance =
				regularized.inv(cv::DECOMP_SVD);
			dataGroup.pose_covariance_trace = static_cast<float>(
				cv::trace(cv::Mat(covariance))[0]);
			dataGroup.pose_hessian_valid =
				std::isfinite(dataGroup.hessian_condition) &&
				std::isfinite(dataGroup.pose_covariance_trace);
		}
	}
	const char *teacherStrideText = std::getenv("BIT_POSE_TEACHER_STRIDE");
	const int teacherStride = teacherStrideText ? std::atoi(teacherStrideText) : 0;
	if (teacherStride > 0 && currData->index % teacherStride == 0)
	{
		const cv::Matx44f nominalPose = currPose;
		cv::Matx44f nominalMetricPose = nominalPose;
		NormalizePoseByPD(nominalMetricPose);
		const float rotationStep = 3.0f * static_cast<float>(CV_PI) / 180.0f;
		const float translationStep = 0.005f;
		double rotationSquared = 0.0;
		double translationSquared = 0.0;
		for (int axis = 0; axis < 6; ++axis)
		{
			for (int sign : {-1, 1})
			{
				cv::Matx61f perturbation = cv::Matx61f::zeros();
				perturbation(axis, 0) = sign *
					(axis < 3 ? rotationStep : translationStep);
				m_objects[0]->setPose(
					summer::Transformations::exp(perturbation) * nominalPose);
				tracker_->EstimatePoses(currData->frame, 2);
				const cv::Matx44f recovered = m_objects[0]->getPose();
				cv::Matx44f recoveredMetricPose = recovered;
				NormalizePoseByPD(recoveredMetricPose);
				cv::Matx33f nominalRotation, recoveredRotation;
				for (int row = 0; row < 3; ++row)
					for (int col = 0; col < 3; ++col)
					{
						nominalRotation(row, col) = nominalMetricPose(row, col);
						recoveredRotation(row, col) = recoveredMetricPose(row, col);
					}
				const cv::Matx33f relative =
					recoveredRotation * nominalRotation.t();
				const float cosine = std::max(-1.0f, std::min(1.0f,
					(static_cast<float>(cv::trace(cv::Mat(relative))[0]) - 1.0f) / 2.0f));
				const float rotationDeg = std::acos(cosine) * 180.0f /
					static_cast<float>(CV_PI);
				const cv::Vec3f translation(
					recovered(0, 3) - nominalPose(0, 3),
					recovered(1, 3) - nominalPose(1, 3),
					recovered(2, 3) - nominalPose(2, 3));
				const float translationMm = cv::norm(translation) * 1000.0f;
				rotationSquared += rotationDeg * rotationDeg;
				translationSquared += translationMm * translationMm;
				dataGroup.pose_teacher_failures +=
					(rotationDeg > 1.0f || translationMm > 2.0f);
				++dataGroup.pose_teacher_probes;
			}
		}
		m_objects[0]->setPose(nominalPose);
		dataGroup.pose_teacher_valid = true;
		dataGroup.pose_teacher_failure_rate = static_cast<float>(
			dataGroup.pose_teacher_failures) / dataGroup.pose_teacher_probes;
		dataGroup.pose_teacher_rotation_rms_deg = std::sqrt(
			rotationSquared / dataGroup.pose_teacher_probes);
		dataGroup.pose_teacher_translation_rms_mm = std::sqrt(
			translationSquared / dataGroup.pose_teacher_probes);
	}
	const char *teacherAblationText = std::getenv("BIT_POSE_TEACHER_ABLATION");
	const bool teacherAblation = teacherAblationText && std::atoi(teacherAblationText) != 0;
	if (teacherAblation && dataGroup.pose_teacher_valid)
	{
		const cv::Matx44f nominalPose = currPose;
		cv::Matx44f nominalMetricPose = nominalPose;
		NormalizePoseByPD(nominalMetricPose);
		const float rotationSteps[3] = {
			1.0f * static_cast<float>(CV_PI) / 180.0f,
			3.0f * static_cast<float>(CV_PI) / 180.0f,
			5.0f * static_cast<float>(CV_PI) / 180.0f};
		const float translationSteps[3] = {0.002f, 0.005f, 0.010f};
		int axisFailures[3] = {0, dataGroup.pose_teacher_failures, 0};
		int axisProbes[3] = {0, dataGroup.pose_teacher_probes, 0};
		int jointFailures[3] = {0, 0, 0};
		int jointProbes[3] = {0, 0, 0};

		auto probeFailed = [&](const cv::Matx61f &perturbation) {
			m_objects[0]->setPose(
				summer::Transformations::exp(perturbation) * nominalPose);
			tracker_->EstimatePoses(currData->frame, 2);
			const cv::Matx44f recovered = m_objects[0]->getPose();
			cv::Matx44f recoveredMetricPose = recovered;
			NormalizePoseByPD(recoveredMetricPose);
			cv::Matx33f nominalRotation, recoveredRotation;
			for (int row = 0; row < 3; ++row)
				for (int col = 0; col < 3; ++col)
				{
					nominalRotation(row, col) = nominalMetricPose(row, col);
					recoveredRotation(row, col) = recoveredMetricPose(row, col);
				}
			const cv::Matx33f relative = recoveredRotation * nominalRotation.t();
			const float cosine = std::max(-1.0f, std::min(1.0f,
				(static_cast<float>(cv::trace(cv::Mat(relative))[0]) - 1.0f) / 2.0f));
			const float rotationDeg = std::acos(cosine) * 180.0f /
				static_cast<float>(CV_PI);
			const cv::Vec3f translation(
				recovered(0, 3) - nominalPose(0, 3),
				recovered(1, 3) - nominalPose(1, 3),
				recovered(2, 3) - nominalPose(2, 3));
			return rotationDeg > 1.0f || cv::norm(translation) * 1000.0f > 2.0f;
		};

		for (int radius : {0, 2})
			for (int axis = 0; axis < 6; ++axis)
				for (int sign : {-1, 1})
				{
					cv::Matx61f perturbation = cv::Matx61f::zeros();
					perturbation(axis, 0) = sign *
						(axis < 3 ? rotationSteps[radius] : translationSteps[radius]);
					axisFailures[radius] += probeFailed(perturbation);
					++axisProbes[radius];
					++dataGroup.pose_teacher_ablation_probes;
				}

		const int signs[4][6] = {
			{1, 1, 1, 1, 1, 1}, {1, -1, 1, -1, 1, -1},
			{-1, 1, 1, 1, -1, -1}, {1, 1, -1, -1, -1, 1}};
		const float invSqrtThree = 1.0f / std::sqrt(3.0f);
		for (int radius = 0; radius < 3; ++radius)
			for (int pattern = 0; pattern < 4; ++pattern)
			{
				cv::Matx61f perturbation = cv::Matx61f::zeros();
				for (int axis = 0; axis < 3; ++axis)
					perturbation(axis, 0) = signs[pattern][axis] *
						rotationSteps[radius] * invSqrtThree;
				for (int axis = 3; axis < 6; ++axis)
					perturbation(axis, 0) = signs[pattern][axis] *
						translationSteps[radius] * invSqrtThree;
				jointFailures[radius] += probeFailed(perturbation);
				++jointProbes[radius];
				++dataGroup.pose_teacher_ablation_probes;
			}
		m_objects[0]->setPose(nominalPose);
		dataGroup.pose_teacher_axis_failure_rate_small =
			static_cast<float>(axisFailures[0]) / axisProbes[0];
		dataGroup.pose_teacher_axis_failure_rate_medium =
			static_cast<float>(axisFailures[1]) / axisProbes[1];
		dataGroup.pose_teacher_axis_failure_rate_large =
			static_cast<float>(axisFailures[2]) / axisProbes[2];
		dataGroup.pose_teacher_joint_failure_rate_small =
			static_cast<float>(jointFailures[0]) / jointProbes[0];
		dataGroup.pose_teacher_joint_failure_rate_medium =
			static_cast<float>(jointFailures[1]) / jointProbes[1];
		dataGroup.pose_teacher_joint_failure_rate_large =
			static_cast<float>(jointFailures[2]) / jointProbes[2];
	}
	SetTrackingResult(ResultType::kResDataGroup, dataGroup);

	cv::Mat result = renderer_->DrawResultOverlay(std::vector<summer::Model *>(m_objects.begin(), m_objects.end()), currData->frame, true);
	cv::imshow("result", result);

	cv::Rect roi = summer::Renderer::Instance()->Compute2DROI(m_objects[0], currData->frame.size(), 8);

	ar3dv::StatsCollector fail_counter("fail nums");
	if (!success)
	{
		fail_counter.IncrementOne();
		m_failCnts++;
		m_objects[0]->setPose(currData->gt);
		VLOG(0) << "m_failCnts " << m_failCnts << " " << currData->index;
	}

	SetTrackingResult(ResultType::kResIndex, currData->index);
	SetTrackingResult(ResultType::kResPose, currPose);
	SetTrackingResult(ResultType::kResGt, currData->gt);
	SetTrackingResult(ResultType::kResOrigin, currData->frame);
	SetTrackingResult(ResultType::kResProbability, prob);
	SetTrackingResult(ResultType::kResOverlay, result);
	SetTrackingResult(ResultType::kResMask, renderer_->DrawMask(std::vector<summer::Model *>(m_objects.begin(), m_objects.end())));
	SetTrackingResult(ResultType::kResROI, roi);
}

void SummerTracker::Init()
{
	SetModels();
	SetCameras();
	SetZnearZfar();
	renderer_ = summer::Renderer::Instance();
	renderer_->init(m_K, m_width, m_height, m_zn, m_zf, 4);
	tracker_ = summer::Tracker::GetTracker(m_K, m_distCoeffs, m_objects);
	VLOG(0) << "SetTracker Done";
}

void SummerTracker::SetCameras()
{
	m_camParams = SystemBase::Instance()->m_camParams;
	m_fx = m_camParams[0].fx;
	m_fy = m_camParams[0].fy;
	m_cx = m_camParams[0].cx;
	m_cy = m_camParams[0].cy;
	m_K = {m_fx, 0, m_cx, 0, m_fy, m_cy, 0, 0, 1};

	m_width = m_camParams[0].width;
	m_height = m_camParams[0].height;
	VLOG(0) << "m_K " << m_K;
	return;
}

void SummerTracker::SetModels()
{
	m_distCoeffs = cv::Matx14f(0.0, 0.0, 0.0, 0.0);
	m_distances = {200.0f, 400.0f, 600.0f};
	m_models = SystemBase::Instance()->m_models;
	VLOG(0) << "m_models.size() " << m_models.size();
	for (auto m : m_models)
		m_objects.push_back(new summer::Object3D(m.path, cv::Matx44f::eye(), 1.0, 0.55f, m_distances));
	CHECK(m_objects.size() != 0) << "3D Model Can Not Be Loaded!";
	return;
}

void SummerTracker::SetPose(const cv::Matx44f &pose)
{
	m_objects[0]->setPose(pose);
	return;
}

void SummerTracker::PrintInfo(const int &flag)
{
	VLOG(flag) << "cam params " << m_fx << " " << m_fy << " " << m_cx << " " << m_cy << " " << m_width << " " << m_height;
	VLOG(flag) << "m_K " << m_K;
}

void SummerTracker::ChangeModels(const MeshModel &meshModel)
{
	cv::Matx44f pose = m_objects[0]->getPose();

	m_models.clear();
	m_models.push_back(meshModel);
	m_objects.clear();
	for (auto m : m_models)
		m_objects.push_back(new summer::Object3D(m.path, cv::Matx44f::eye(), 1.0, 0.55f, m_distances));
	CHECK(m_objects.size() != 0) << "3D Model Can Not Be Changed!";
	m_objects[0]->setPose(pose);
	if (tracker_)
		delete tracker_;
	tracker_ = summer::Tracker::GetTracker(m_K, m_distCoeffs, m_objects);
	m_objects[0]->setPose(pose);
	tracker_->ToggleTracking(m_stdDatas[0]->frame, 0, false);
	return;
}

cv::Vec2f SummerTracker::ComRoiCenter(const cv::Rect &roi)
{
	float x = float(roi.x) + 0.5f * float(roi.width);
	float y = float(roi.y) + 0.5f * float(roi.height);
	return cv::Vec2f(x, y);
}

cv::Matx33f SummerTracker::ComCenteredScaledIntrinsics(const cv::Matx33f &K, const float &w, const float &h, const cv::Rect &roi, const float &scale)
{
	float x = float(roi.x) + 0.5f * float(roi.width);
	float y = float(roi.y) + 0.5f * float(roi.height);
	cv::Vec2f currCenter(x, y);
	cv::Matx33f res = cv::Matx33f::eye();
	float delta_cx = float(currCenter[0] * scale - 0.5f * w); // 0.5w is the center of image
	float delta_cy = float(currCenter[1] * scale - 0.5f * h);
	res(0, 0) = K(0, 0) * scale;
	res(1, 1) = K(1, 1) * scale;
	res(0, 2) = K(0, 2) * scale - delta_cx;
	res(1, 2) = K(1, 2) * scale - delta_cy;
	return res;
}

void SummerTracker::ConvertToThreeChannels(cv::Mat &input)
{
	cv::Mat output(input.rows, input.cols, CV_8UC3, cv::Scalar(0, 0, 0));
	cv::Mat in[] = {input, input, input};
	int from_to[] = {0, 0, 1, 1, 2, 2};
	cv::mixChannels(in, 3, &output, 1, from_to, 3);
	input = output;
	return;
}

DataGroup SummerTracker::ComDataGroup(const cv::Mat &frame, const cv::Mat &prob, const bool &genRefer)
{
	DataGroup dataGroup;
	dataGroup.valid = false;
	Eigen::Vector3f defaultView(0, 0, 0);
	dataGroup.view = defaultView;

	float w = (float)frame.cols;
	float h = (float)frame.rows;
	float expansionRatio = 1.05;
	cv::Vec2f frameCenter(0.5 * w, 0.5 * h);
	cv::Matx33f currK = renderer_->K33(0);

	cv::Rect roi = renderer_->RoiByMask();
	tracker_->prompter()->ExpandRoi(roi, expansionRatio);
	if (!tracker_->prompter()->CheckRoi(roi, w, h, 10))
	{
		return dataGroup;
	}

	float scale = tracker_->prompter()->ComScaleFactor((int)w, (int)h, roi.width, roi.height, 0.9f);
	cv::Matx33f KScaled = ComCenteredScaledIntrinsics(currK, w, h, roi, scale);

	renderer_->ChangeK44s(KScaled, m_zn, m_zf);
	renderer_->setLevel(0);
	renderer_->RenderShaded(std::vector<summer::Model *>(m_objects.begin(), m_objects.end()), GL_FILL);
	cv::Rect roi_scaled_centered = renderer_->RoiByMask();
	tracker_->prompter()->ExpandRoi(roi_scaled_centered, expansionRatio);

	int target_w = w / scale;
	int target_h = h / scale;
	cv::Mat prob_target = cv::Mat::zeros(target_h, target_w, prob.type());
	cv::Rect targetRoi(roi_scaled_centered.x / scale, roi_scaled_centered.y / scale, roi.width, roi.height);
	if (!tracker_->prompter()->CheckRoi(targetRoi, w, h, 0))
	{
		renderer_->ChangeK44s(currK, m_zn, m_zf);
		return dataGroup;
	}

	prob(roi).copyTo(prob_target(targetRoi));

	cv::Mat mask;
	if (genRefer)
		mask = prob.clone();
	else
		mask = tracker_->prompter()->ComConfidenceMaskFromProb(prob_target, 32, 16);

	cv::resize(mask, mask, cv::Size(w, h));
	cv::resize(prob_target, prob_target, cv::Size(w, h));
	ConvertToThreeChannels(prob_target);

	renderer_->ChangeK44s(currK, m_zn, m_zf);

	dataGroup.valid = true;
	dataGroup.prob = prob_target;
	dataGroup.mask = mask;
	dataGroup.img = frame.clone();
	dataGroup.K = KScaled;
	dataGroup.pose = m_objects[0]->getPose();
	dataGroup.originRoi = roi;
	dataGroup.targetRoi = targetRoi;
	dataGroup.targetSize = cv::Vec2f(target_w, target_h);
	dataGroup.targetWidth = target_w;
	dataGroup.targetHeight = target_h;

	cv::Matx44f T = m_objects[0]->getPose();
	cv::Matx33f R(T(0, 0), T(0, 1), T(0, 2),
				  T(1, 0), T(1, 1), T(1, 2),
				  T(2, 0), T(2, 1), T(2, 2));
	cv::Vec3f t(T(0, 3), T(1, 3), T(2, 3));
	float norm_t = cv::norm(t);
	t /= norm_t;
	cv::Vec3f viewCV = R.inv() * t;
	Eigen::Vector3f view(viewCV[0], viewCV[1], viewCV[2]);

	dataGroup.view = view;
	return dataGroup;
}
