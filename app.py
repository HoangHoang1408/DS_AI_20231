import json
import os
from random import choices

import faiss
import gradio as gr
import joblib
import numpy as np

base_path = "/Users/phamhoang1408/Desktop/20231/DS/ds_project/models"
with open(f"{base_path}/features/cap_bac.json", "r") as f:
    cap_bac_feature_values = json.load(f)
with open(f"{base_path}/features/dia_diem_lam_viec.json", "r") as f:
    dia_diem_lam_viec_feature_values = json.load(f)
with open(f"{base_path}/features/hinh_thuc.json", "r") as f:
    hinh_thuc_feature_values = json.load(f)
with open(f"{base_path}/features/loai_hinh_hoat_dong.json", "r") as f:
    loai_hinh_hoat_dong_feature_values = json.load(f)
with open(f"{base_path}/features/nganh_nghe.json", "r") as f:
    nganh_nghe_feature_values = json.load(f)
with open(f"{base_path}/features/quy_mo_cong_ty.json", "r") as f:
    quy_mo_cong_ty_feature_values = json.load(f)
with open(f"{base_path}/features/ten_cong_ty.json", "r") as f:
    ten_cong_ty_feature_values = json.load(f)

with open(f"{base_path}/features/num_followers.json", "r") as f:
    num_followers_feature_values = json.load(f)

vi_tri_viec_vectorizer = joblib.load(
    f"{base_path}/features/vi_tri_viec_vectorizer.joblib"
)


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


model_min = joblib.load(f"{base_path}/models/lgbm_min.joblib")
model_max = joblib.load(f"{base_path}/models/lgbm_max.joblib")


def get_prediction(data):
    feature_vector = convert_raw_data_to_feature(data)
    min_label_to_range = {
        0: (0, 5),
        1: (5, 10),
        2: (10, 15),
        3: (15, 20),
        4: (20, 25),
        5: (25, 30),
        6: (30, 50),
        7: (50, 75),
        8: (75, 100),
        9: (100, 999999),
    }
    max_label_to_range = {
        0: (0, 10),
        1: (10, 20),
        2: (20, 30),
        3: (30, 50),
        4: (50, 75),
        5: (75, 100),
        6: (100, 999999),
    }
    min_pred = model_min.predict(feature_vector.reshape(1, -1))[0]
    max_pred = model_max.predict(feature_vector.reshape(1, -1))[0]
    min_range = min_label_to_range[min_pred]
    max_range = max_label_to_range[max_pred]
    print(min_range, max_range)
    min_ = min_range[0]
    max_ = max_range[0]
    return min_, max_


base_path = "/Users/phamhoang1408/Desktop/20231/DS/ds_project/models"
embeddings = np.load(f"{base_path}/embeddings.npy")
index = faiss.IndexFlatIP(embeddings.shape[1])


def pred_salary(
    vi_tri_viec,
    cap_bac,
    dia_diem_lam_viec,
    hinh_thuc,
    loai_hinh_hoat_dong,
    nganh_nghe,
    quy_mo_cong_ty,
    ten_cong_ty,
    num_followers,
):
    print(
        vi_tri_viec,
        cap_bac,
        dia_diem_lam_viec,
        hinh_thuc,
        loai_hinh_hoat_dong,
        nganh_nghe,
        quy_mo_cong_ty,
        ten_cong_ty,
        num_followers,
    )
    min_, max_ = get_prediction(
        {
            "vi_tri_viec": vi_tri_viec,
            "cap_bac": cap_bac,
            "dia_diem_lam_viec": dia_diem_lam_viec,
            "hinh_thuc": hinh_thuc,
            "loai_hinh_hoat_dong": loai_hinh_hoat_dong,
            "nganh_nghe": nganh_nghe,
            "quy_mo_cong_ty": quy_mo_cong_ty,
            "ten_cong_ty": ten_cong_ty,
            "num_followers": num_followers,
        }
    )
    return min_, max_


demo = gr.Blocks(title="Salary Estimator")
with demo:
    with gr.Tabs():
        with gr.TabItem("Salary Estimator"):
            inputs_tab1 = [
                vi_tri_viec_textbox := gr.Textbox(label="Vị trí việc"),
                cap_bac_dropdown := gr.Dropdown(
                    choices=[x for x in cap_bac_feature_values if x != None],
                    label="Cấp bậc",
                ),
                dia_diem_lam_viec_dropdown := gr.Dropdown(
                    choices=[x for x in dia_diem_lam_viec_feature_values if x != None],
                    label="Địa điểm làm việc",
                    allow_custom_value=True,
                ),
                hinh_thuc_dropdown := gr.Dropdown(
                    choices=[x for x in hinh_thuc_feature_values if x != None],
                    label="Hình thức",
                ),
                loai_hinh_hoat_dong_dropdown := gr.Dropdown(
                    choices=[
                        x for x in loai_hinh_hoat_dong_feature_values if x != None
                    ],
                    label="Loại hình hoạt động",
                    allow_custom_value=True,
                ),
                nganh_nghe_dropdown := gr.Dropdown(
                    choices=[x for x in nganh_nghe_feature_values if x != None],
                    label="Ngành nghề",
                    allow_custom_value=True,
                ),
                quy_mo_cong_ty_number := gr.Number(label="Quy mô công ty"),
                ten_cong_ty_dropdown := gr.Dropdown(
                    choices=[x for x in ten_cong_ty_feature_values if x != None],
                    label="Tên công ty",
                    allow_custom_value=True,
                ),
                num_followers_number := gr.Number(label="Số lượng người theo dõi"),
            ]
            btn1 = gr.Button(value="Dự đoán")
            outputs_tab1 = [
                gr.Label(label="Lương tối thiểu"),
                gr.Label(label="Lương tối đa"),
            ]
        with gr.TabItem("Search Job"):
            inputs_tab2 = [gr.Text(label="Tìm kiếm việc làm")]
            btn2 = gr.Button(value="Tìm kiếm")
            outputs_tab2 = [gr.Text()]
    btn1.click(pred_salary, inputs=inputs_tab1, outputs=outputs_tab1)
    btn2.click(pred_salary, inputs=inputs_tab2, outputs=outputs_tab2)

demo.launch()
