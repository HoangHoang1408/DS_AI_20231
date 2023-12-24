import json
from time import sleep

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

service = Service(
    "/Users/phamhoang1408/Desktop/20231/DS/ds_project/crawl/driver/chromedriver"
)
option = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=service, options=option)

data = []
i = 200
while True:
    try:
        i += 1
        driver.get(
            f"https://careerbuilder.vn/viec-lam/tat-ca-viec-lam-trang-{i}-vi.html"
        )
        print(i)
        page_job_links = driver.find_elements(
            By.CSS_SELECTOR,
            "div.job-item",
        )
        for page_job_link in page_job_links:
            job_link = page_job_link.find_element(
                By.CSS_SELECTOR,
                "a.job_link",
            ).get_attribute("href")
            data.append({"job_link": job_link})
        # next_page = driver.find_element(By.CSS_SELECTOR, "li.next-page").find_element(
        #     By.CSS_SELECTOR, "a"
        # )
        # next_page.click()
        if i % 10 == 0:
            print("Saving...")
            with open(
                f"/Users/phamhoang1408/Desktop/20231/DS/ds_project/crawl/job_links/job_to_page{i}.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(data, f, ensure_ascii=False)
            data = []
    except Exception as e:
        print(e)
        driver.close()
