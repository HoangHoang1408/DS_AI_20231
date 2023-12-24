import json
import os
from glob import glob
from pprint import pprint
from time import sleep

import numpy as np
import pandas as pd
from datasets import Dataset
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

service = Service(
    "/Users/phamhoang1408/Desktop/20231/DS/ds_project/crawl/driver/chromedriver"
)
option = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=service, options=option)

links = Dataset.from_json(
    "/Users/phamhoang1408/Desktop/20231/DS/ds_project/crawl/data2/jobs.jsonl"
)["job_link"]
print(len(links))
data = []
start = 0
print(start)
for start, link in enumerate(links[start], start=start):
    d = {}
    d["link"] = link
    try:
        driver.get(link)
        temp1 = driver.find_element(By.CSS_SELECTOR, "div.job-desc")
        d["vi_tri_viec"] = temp1.find_element(By.CSS_SELECTOR, "h1.title").text
        temp2 = temp1.find_element(By.CSS_SELECTOR, "a.employer")
        d["ten_cong_ty"] = temp2.text
        d["company_link"] = temp2.get_attribute("href")
        detail_boxes = driver.find_elements(By.CSS_SELECTOR, "div.detail-box")
        for detail_box in detail_boxes:
            strongs = detail_box.find_elements(By.CSS_SELECTOR, "strong")
            if len(strongs) == 1:
                strong = strongs[0]
                d[strong.text] = (
                    detail_box.find_element(By.CSS_SELECTOR, "p")
                    .find_element(By.CSS_SELECTOR, "a")
                    .text
                )
            else:
                lis = detail_box.find_elements(By.CSS_SELECTOR, "li")
                for li in lis:
                    title = li.find_element(By.CSS_SELECTOR, "strong").text
                    value = li.find_element(By.CSS_SELECTOR, "p").text
                    d[title] = value
        tab_company = driver.find_element(By.CSS_SELECTOR, "li#tabs-job-company")
        tab_company.click()
        sleep(2)
        company_info = driver.find_element(
            By.CSS_SELECTOR, "div.company-introduction"
        ).find_element(By.CSS_SELECTOR, "div.content")
        d["company_info"] = company_info.text
        follower_number = (
            driver.find_element(By.CSS_SELECTOR, "div.company-follow")
            .find_element(By.CSS_SELECTOR, "strong")
            .text
        )
        d["company_follower_number"] = follower_number
        d["error"] = False

    except Exception as e:
        d["error"] = True

    data.append(d)
    if (start + 1) % 10 == 0:
        with open(
            f"/Users/phamhoang1408/Desktop/20231/DS/ds_project/crawl/job_details/data_{start}.json",
            "w",
        ) as f:
            json.dump(data, f)
        data = []
driver.close()
