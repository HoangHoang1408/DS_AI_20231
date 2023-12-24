import json

import joblib
import numpy as np

base_path = "/Users/phamhoang1408/Desktop/20231/DS/ds_project/models/features"
with open(f"{base_path}/cap_bac.json", "r") as f:
    cap_bac_feature_values = json.load(f)
with open(f"{base_path}/dia_diem_lam_viec.json", "r") as f:
    dia_diem_lam_viec_feature_values = json.load(f)
with open(f"{base_path}/hinh_thuc.json", "r") as f:
    hinh_thuc_feature_values = json.load(f)
with open(f"{base_path}/loai_hinh_hoat_dong.json", "r") as f:
    loai_hinh_hoat_dong_feature_values = json.load(f)
with open(f"{base_path}/nganh_nghe.json", "r") as f:
    nganh_nghe_feature_values = json.load(f)
with open(f"{base_path}/quy_mo_cong_ty.json", "r") as f:
    quy_mo_cong_ty_feature_values = json.load(f)
with open(f"{base_path}/ten_cong_ty.json", "r") as f:
    ten_cong_ty_feature_values = json.load(f)

with open(f"{base_path}/num_followers.json", "r") as f:
    num_followers_feature_values = json.load(f)

vi_tri_viec_vectorizer = joblib.load(f"{base_path}/vi_tri_viec_vectorizer.joblib")


def convert_raw_data_to_feature(raw_data):
    def convert_text_unique_to_feature(feature_values, value):
        vector = np.zeros(len(feature_values))
        if value in feature_values:
            vector[feature_values.index(value)] = 1
        elif None in feature_values:
            vector[-1] = 1
        return vector

    def convert_numeric_to_feature(ranges, value):
        vector = np.zeros(len(ranges))
        if value is None:
            vector[-1] = 1
        else:
            for i, r in enumerate(ranges):
                if r[0] <= value <= r[1]:
                    vector[i] = 1
                    break
        return vector

    def convert_tf_idf_to_feature(vectorizer, value):
        return vectorizer.transform([value]).toarray()[0]

    feature_vector = np.concatenate(
        [
            convert_text_unique_to_feature(cap_bac_feature_values, raw_data["cap_bac"]),
            convert_text_unique_to_feature(
                dia_diem_lam_viec_feature_values, raw_data["dia_diem_lam_viec"]
            ),
            convert_text_unique_to_feature(
                hinh_thuc_feature_values, raw_data["hinh_thuc"]
            ),
            convert_text_unique_to_feature(
                loai_hinh_hoat_dong_feature_values, raw_data["loai_hinh_hoat_dong"]
            ),
            convert_text_unique_to_feature(
                nganh_nghe_feature_values, raw_data["nganh_nghe"]
            ),
            convert_text_unique_to_feature(
                quy_mo_cong_ty_feature_values, raw_data["quy_mo_cong_ty"]
            ),
            convert_text_unique_to_feature(
                ten_cong_ty_feature_values, raw_data["ten_cong_ty"]
            ),
            convert_tf_idf_to_feature(vi_tri_viec_vectorizer, raw_data["vi_tri_viec"]),
            convert_numeric_to_feature(
                num_followers_feature_values, raw_data["num_followers"]
            ),
        ]
    )
    return feature_vector
