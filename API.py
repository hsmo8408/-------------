import pandas as pd
import requests
import json
import argparse
from collections import defaultdict


import os
import logging
from datetime import datetime
import traceback
import pandas as pd
import urllib.parse
from sqlalchemy import create_engine


KAKAO_API_KEY = 'c40dfb29df262a833ba6a474563319bb'
address_cache = {} 

def get_kakao_address_data(address):
    if address in address_cache:
        return address_cache[address]

    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    params = {"query": address}

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        result = response.json()
        if result['documents']:
            document = result['documents'][0]
            x = document['x']
            y = document['y']
            sigungu_code = document['address']['b_code'][:5]  
            address_cache[address] = (x, y, sigungu_code)  
            return x, y, sigungu_code
    return None, None, None

def add_coordinates_and_code(data, address_column):
    # 모든 주소에 대해 좌표와 시군구 코드 가져오기
    results = data[address_column].apply(get_kakao_address_data)
    data['x_coord'] = results.apply(lambda x: x[0])
    data['y_coord'] = results.apply(lambda x: x[1])
    data['sigungu_code'] = results.apply(lambda x: x[2])

    return data

class LoggerFactory(object):
    _LOGGER = None
    
    @staticmethod
    def create_logger():
        LoggerFactory._LOGGER = logging.getLogger()
        LoggerFactory._LOGGER.setLevel(logging.INFO)
        if not os.path.exists('./papercompany_log'):
            os.makedirs('./papercompany_log')                          
        formatter = logging.Formatter('[%(asctime)s][ggd_papercompany_log|%(funcName)s:%(lineno)s] >> %(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        file_handler = logging.FileHandler('./papercompany_log/' + datetime.now().strftime('%Y%m') + '.log')
        file_handler.setFormatter(formatter)
        LoggerFactory._LOGGER.addHandler(stream_handler)
        LoggerFactory._LOGGER.addHandler(file_handler)

    @classmethod
    def get_logger(cls):
        return cls._LOGGER
    
LoggerFactory.create_logger()


### 패키지 불러오기 ###
try:
    import numpy as np
    import re
    import pickle
    import geopandas as gpd
    from pyproj import Transformer
    from pyproj import transform
    from sklearn.preprocessing import LabelEncoder
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    from imblearn.over_sampling import SMOTE
except Exception as e:
    error_code = traceback.format_exc()
    LoggerFactory.get_logger().info("Packages Loading Error" + " || Error is :: " + error_code)
finally:
    if 'error_code' in locals():
        del(error_code)
    else:
        LoggerFactory.get_logger().info("Packages Loading Complete")

### Create engine ###
try:
    password = urllib.parse.quote('spdev1234!@#')
    db_engine = create_engine(f'postgresql://postgres:{password}@112.220.90.92:5432/db_anal')  # Change the database as needed
    LoggerFactory.get_logger().info("Database engine created successfully.")
except Exception as e:
    error_code = traceback.format_exc()
    LoggerFactory.get_logger().info("Create engine Error" + " || Error is :: " + error_code)
finally:
    if 'error_code' in locals():
        del(error_code)

### 업체정보 데이터 로드 ###
query = "SELECT * FROM company.tb_crack_down_biz_1112"

try:
    c_biz = pd.read_sql(query, con=db_engine)
    LoggerFactory.get_logger().info("Data loading complete, DataFrame shape: {}".format(c_biz.shape))
except Exception as e:
    error_code = traceback.format_exc()
    LoggerFactory.get_logger().info("Data loading Error" + " || Error is :: " + error_code)
finally:
    if 'error_code' in locals():
        del(error_code)

### 업종정보 데이터 로드 ###
query = "SELECT * FROM company.tb_crack_down_industry_1112"

try:
    c_ind = pd.read_sql(query, con=db_engine)
    LoggerFactory.get_logger().info("Data loading complete, DataFrame shape: {}".format(c_ind.shape))
except Exception as e:
    error_code = traceback.format_exc()
    LoggerFactory.get_logger().info("Data loading Error" + " || Error is :: " + error_code)
finally:
    if 'error_code' in locals():
        del(error_code)



### 행정처분 데이터 로드 ###
query = "SELECT * FROM company.tb_admi"

try:
    row_admi = pd.read_sql(query, con=db_engine)
    LoggerFactory.get_logger().info("Data loading complete, DataFrame shape: {}".format(row_admi.shape))
except Exception as e:
    error_code = traceback.format_exc()
    LoggerFactory.get_logger().info("Data loading Error" + " || Error is :: " + error_code)
finally:
    if 'error_code' in locals():
        del(error_code)
        
### 폐업신고 데이터 로드 ###
query = "SELECT * FROM company.tb_cess"

try:
    row_cess = pd.read_sql(query, con=db_engine)
    LoggerFactory.get_logger().info("Data loading complete, DataFrame shape: {}".format(row_cess.shape))
except Exception as e:
    error_code = traceback.format_exc()
    LoggerFactory.get_logger().info("Data loading Error" + " || Error is :: " + error_code)
finally:
    if 'error_code' in locals():
        del(error_code)


def main(sDate, eDate):
    # GongsiRenew
    try:
        con_df = pd.DataFrame()
        for page_nums in range(1, 10): 
            url = f"http://apis.data.go.kr/1613000/ConAdminInfoSvc1/GongsiRenew"
            params = {
                'pageNo': page_nums,
                'numOfRows': 10000000,
                'sDate': sDate,  
                'eDate': eDate,
                'ncrAreaName': '경기',
                '_type': 'json',
                'serviceKey': '0/GzGqodwWbM6G31J5WiK9Gk1Cg0vnzeNwW+fOvyfrXDqohWYLra8DZX4pLGpw4jNRsBKSn4Vfgr11x19JNgyg=='
            }
            response = requests.get(url, params=params)
            response.raise_for_status()

            if not response.text:
                raise json.decoder.JSONDecodeError("Empty response", response.text, 0)

            data = response.json()
            if data['response']['body']['items'] != '':
                df = pd.DataFrame(data['response']['body']['items']['item'])
                con_df = pd.concat([con_df, df])
            else:
                break
            

    except json.decoder.JSONDecodeError as e:
        print(f"JSON 디코딩 오류: {e}")
    except requests.exceptions.RequestException as e:
        print(f"요청 예외: {e}")
    except Exception as e:
        print(f"기타 오류: {e}")

    # GongsiTrans
    try:
        con_df = pd.DataFrame()
        for page_nums in range(1, 10): 
            url = f"http://apis.data.go.kr/1613000/ConAdminInfoSvc1/GongsiTrans"
            params = {
                'pageNo': page_nums,
                'numOfRows': 10000000,
                'sDate': sDate,  
                'eDate': eDate,
                'ncrAreaName': '경기',
                '_type': 'json',
                'serviceKey': '0/GzGqodwWbM6G31J5WiK9Gk1Cg0vnzeNwW+fOvyfrXDqohWYLra8DZX4pLGpw4jNRsBKSn4Vfgr11x19JNgyg=='
            }
            response = requests.get(url, params=params)
            response.raise_for_status()

            if not response.text:
                raise json.decoder.JSONDecodeError("Empty response", response.text, 0)

            data = response.json()
            if data['response']['body']['items'] != '':
                df = pd.DataFrame(data['response']['body']['items']['item'])
                con_df = pd.concat([con_df, df])
            else:
                break


    except json.decoder.JSONDecodeError as e:
        print(f"JSON 디코딩 오류: {e}")
    except requests.exceptions.RequestException as e:
        print(f"요청 예외: {e}")
    except Exception as e:
        print(f"기타 오류: {e}")

    # GongsiUnion
    try:
        con_df = pd.DataFrame()
        for page_nums in range(1, 10): 
            url = f"http://apis.data.go.kr/1613000/ConAdminInfoSvc1/GongsiUnion"
            params = {
                'pageNo': page_nums,
                'numOfRows': 10000000,
                'sDate': sDate,  
                'eDate': eDate,
                'ncrAreaName': '경기',
                '_type': 'json',
                'serviceKey': '0/GzGqodwWbM6G31J5WiK9Gk1Cg0vnzeNwW+fOvyfrXDqohWYLra8DZX4pLGpw4jNRsBKSn4Vfgr11x19JNgyg=='
            }
            response = requests.get(url, params=params)
            response.raise_for_status()

            if not response.text:
                raise json.decoder.JSONDecodeError("Empty response", response.text, 0)

            data = response.json()
            if data['response']['body']['items'] != '':
                df = pd.DataFrame(data['response']['body']['items']['item'])
                con_df = pd.concat([con_df, df])
            else:
                break

    except json.decoder.JSONDecodeError as e:
        print(f"JSON 디코딩 오류: {e}")
    except requests.exceptions.RequestException as e:
        print(f"요청 예외: {e}")
    except Exception as e:
        print(f"기타 오류: {e}")

    # GongsiInheri
    try:
        con_df = pd.DataFrame()
        for page_nums in range(1, 10): 
            url = f"http://apis.data.go.kr/1613000/ConAdminInfoSvc1/GongsiInheri"
            params = {
                'pageNo': page_nums,
                'numOfRows': 10000000,
                'sDate': sDate,  
                'eDate': eDate,
                'ncrAreaName': '경기',
                '_type': 'json',
                'serviceKey': '0/GzGqodwWbM6G31J5WiK9Gk1Cg0vnzeNwW+fOvyfrXDqohWYLra8DZX4pLGpw4jNRsBKSn4Vfgr11x19JNgyg=='
            }
            response = requests.get(url, params=params)
            response.raise_for_status()

            if not response.text:
                raise json.decoder.JSONDecodeError("Empty response", response.text, 0)

            data = response.json()
            if data['response']['body']['items'] != '':
                df = pd.DataFrame(data['response']['body']['items']['item'])
                con_df = pd.concat([con_df, df])
            else:
                break

    except json.decoder.JSONDecodeError as e:
        print(f"JSON 디코딩 오류: {e}")
    except requests.exceptions.RequestException as e:
        print(f"요청 예외: {e}")
    except Exception as e:
        print(f"기타 오류: {e}")


    # GongsiAdmi
    try:
        con_df = pd.DataFrame()
        for page_nums in range(1, 10): 
            url = f"http://apis.data.go.kr/1613000/ConAdminInfoSvc1/GongsiAdmi"
            params = {
                'pageNo': page_nums,
                'numOfRows': 10000000,
                'sDate': sDate,  
                'eDate': eDate,
                'ncrAreaName': '경기',
                '_type': 'json',
                'serviceKey': '0/GzGqodwWbM6G31J5WiK9Gk1Cg0vnzeNwW+fOvyfrXDqohWYLra8DZX4pLGpw4jNRsBKSn4Vfgr11x19JNgyg=='
            }
            response = requests.get(url, params=params)
            response.raise_for_status()

            if not response.text:
                raise json.decoder.JSONDecodeError("Empty response", response.text, 0)

            data = response.json()
            if data['response']['body']['items'] != '':
                df = pd.DataFrame(data['response']['body']['items']['item'])
                con_df = pd.concat([con_df, df])
            else:
                break

        admi_df = con_df
        
        new_admi_df = pd.concat([row_admi,admi_df], ignore_index=False)
        new_admi_df = new_admi_df.drop_duplicates()

        # 행정처분 db insert
        new_admi_df.to_sql(name='tb_admi', con=db_engine, if_exists='replace', schema='company', index=False)

    except json.decoder.JSONDecodeError as e:
        print(f"JSON 디코딩 오류: {e}")
    except requests.exceptions.RequestException as e:
        print(f"요청 예외: {e}")
    except Exception as e:
        print(f"기타 오류: {e}")

    cess_df = pd.DataFrame()  # 이 줄을 추가합니다.

    # GongsiCess
    try:
        con_df = pd.DataFrame()
        for page_nums in range(1, 10): 
            url = f"http://apis.data.go.kr/1613000/ConAdminInfoSvc1/GongsiCess"
            params = {
                'pageNo': page_nums,
                'numOfRows': 10000000,
                'sDate': sDate,  
                'eDate': eDate,
                'ncrAreaName': '경기',
                '_type': 'json',
                'serviceKey': '0/GzGqodwWbM6G31J5WiK9Gk1Cg0vnzeNwW+fOvyfrXDqohWYLra8DZX4pLGpw4jNRsBKSn4Vfgr11x19JNgyg=='
            }
            response = requests.get(url, params=params)
            response.raise_for_status()

            if not response.text:
                raise json.decoder.JSONDecodeError("Empty response", response.text, 0)

            data = response.json()
            if data['response']['body']['items'] != '':
                df = pd.DataFrame(data['response']['body']['items']['item'])
                con_df = pd.concat([con_df, df])
            else:
                break

        cess_df = con_df
        cess_df['ncrMasterNum'] = cess_df['ncrMasterNum'].astype(float)
        new_cess_df = pd.concat([row_cess,cess_df], ignore_index=False)
        new_cess_df = new_cess_df.drop_duplicates()

        # 폐업신고 정보 db insert
        new_cess_df.to_sql(name='tb_cess', con=db_engine, if_exists='replace', schema='company', index=False)


    except json.decoder.JSONDecodeError as e:
        print(f"JSON 디코딩 오류: {e}")
    except requests.exceptions.RequestException as e:
        print(f"요청 예외: {e}")
    except Exception as e:
        print(f"기타 오류: {e}")

    # GongsiReg
    try:
        con_df = pd.DataFrame()
        for page_nums in range(1, 10): 
            url = f"http://apis.data.go.kr/1613000/ConAdminInfoSvc1/GongsiReg"
            params = {
                'pageNo': page_nums,
                'numOfRows': 10000000,
                'sDate': sDate,  
                'eDate': eDate,
                'ncrAreaName': '경기',
                '_type': 'json',
                'serviceKey': '0/GzGqodwWbM6G31J5WiK9Gk1Cg0vnzeNwW+fOvyfrXDqohWYLra8DZX4pLGpw4jNRsBKSn4Vfgr11x19JNgyg=='
            }
            response = requests.get(url, params=params)
            response.raise_for_status()

            if not response.text:
                raise json.decoder.JSONDecodeError("Empty response", response.text, 0)

            data = response.json()
            if data['response']['body']['items'] != '':
                df = pd.DataFrame(data['response']['body']['items']['item'])
                con_df = pd.concat([con_df, df])
            else:
                break

        reg_df = con_df
        reg_df['ncrMasterNum'] = reg_df['ncrMasterNum'].astype(float)
        # 좌표와 시군구 코드 추가
        reg_df = add_coordinates_and_code(reg_df, 'ncrGsAddr')

        # 경기도 시군구 코드와 이름 매핑 테이블 (시 단위만 포함)
        mapping = {
            '41111' :'수원시 장안구',
            '41113'	:'수원시 권선구',
            '41115'	:'수원시 팔달구',
            '41117'	:'수원시 영통구',
            '41131'	:'성남시 수정구',
            '41133'	:'성남시 중원구',
            '41135'	:'성남시 분당구',
            '41150'	:'의정부시',
            '41171'	:'안양시 만안구',
            '41173'	:'안양시 동안구',
            '41190'	:'부천시',
            '41210'	:'광명시',
            '41220'	:'평택시',
            '41250'	:'동두천시',
            '41271'	:'안산시 상록구',
            '41273'	:'안산시 단원구',
            '41281'	:'고양시 덕양구',
            '41285'	:'고양시 일산동구',
            '41287'	:'고양시 일산서구',
            '41290'	:'과천시',
            '41310'	:'구리시',
            '41360'	:'남양주시',
            '41370'	:'오산시',
            '41390'	:'시흥시',
            '41410'	:'군포시',
            '41430'	:'의왕시',
            '41450'	:'하남시',
            '41461'	:'용인시 처인구',
            '41463'	:'용인시 기흥구',
            '41465'	:'용인시 수지구',
            '41480'	:'파주시',
            '41500'	:'이천시',
            '41550'	:'안성시',
            '41570'	:'김포시',
            '41590'	:'화성시',
            '41610'	:'광주시',
            '41630'	:'양주시',
            '41650'	:'포천시',
            '41670'	:'여주시',
            '41800'	:'연천군',
            '41820'	:'가평군',
            '41830'	:'양평군',
            '41192' :'부천시 원미구',
            '41194' :'부천시 소사구',
            '41196' :'부천시 오정구',
            }

        # 매핑 테이블을 데이터프레임으로 변환
        mapping_df = pd.DataFrame(list(mapping.items()), columns=['sigungu_code', 'sigungu_nm'])

        # 데이터 타입을 문자열로 변환
        reg_df['sigungu_code'] = reg_df['sigungu_code'].astype(str)
        mapping_df['sigungu_code'] = mapping_df['sigungu_code'].astype(str)

        # 원본 데이터와 매핑 데이터 결합
        reg_df = pd.merge(reg_df, mapping_df, on='sigungu_code', how='left')

        # total 및 spec 카테고리 정의
        total = ['토목공사', '건축공사', '토목건축공사', '산업ㆍ환경설비공사', '조경공사']
        spec = [
            '지반조성ㆍ포장공사', '토공사', '포장공사', '보링ㆍ그라우팅공사', '실내건축', 
            '금속ㆍ창호ㆍ지붕ㆍ건축물조립공사업', '금속구조물ㆍ창호ㆍ온실공사업', '지붕판금ㆍ건축물조립공사업', 
            '도장ㆍ습식ㆍ방수ㆍ석공사업', '도장공사', '습식ㆍ방수공사', '석공사', '조경식재ㆍ시설물공사업', 
            '조경식재공사', '조경시설물설치공사', '철근ㆍ콘크리트공사', '구조물해체ㆍ비계공사', '상ㆍ하수도설비공사', 
            '철도ㆍ궤도공사', '철강구조물공사', '수중공사업', '준설공사', '승강기설치공사', '승강기ㆍ삭도공사', 
            '기계설비ㆍ가스공사', '기계설비공사', '가스시설시공업 제1종', '가스시설시공업 제2종', 
            '가스시설시공업 제3종', '가스ㆍ난방공사업', '가스ㆍ난방공사', '난방시공업 제1종', '난방시공업 제2종', 
            '난방시공업 제3종', '시설물유지관리','산업설비공사','철강재설치공사','철물공사','구조물해체공사','삭도설치공사','강구조물공사','방수공사','창호공사','판금공사'
        ]

        # 종합/전문 구분
        def classify_row(ncrItemName):
            total_items = [item for item in total if item in ncrItemName]
            spec_items = [item for item in spec if item in ncrItemName]
            
            induty_mid_inte = '종합' if total_items else 'None'
            induty_mid_spe = '전문' if spec_items else 'None'
            induty_mid_inte_sub = '|'.join(total_items) if total_items else 'None'
            induty_mid_spe_sub = '|'.join(spec_items) if spec_items else 'None'
            
            return pd.Series([induty_mid_inte, induty_mid_spe, induty_mid_inte_sub, induty_mid_spe_sub])

        # 새로운 컬럼 생성
        reg_df[['induty_mid_inte', 'induty_mid_spe', 'induty_mid_inte_sub', 'induty_mid_spe_sub']] = reg_df['ncrItemName'].apply(classify_row)

        # 종합전문 구분 함수
        def com_spec_code(row):
            total_status = row['induty_mid_inte']
            spec_status = row['induty_mid_spe']
            
            if total_status == '종합' and spec_status == '전문':
                return '3'
            elif total_status == '종합':
                return '1'
            elif spec_status == '전문':
                return '2'
            else:
                return None

        reg_df['se_cd'] = reg_df.apply(com_spec_code, axis=1)
        reg_df['open_status'] = '영업중'

        reg_df.columns = reg_df.columns.str.lower()

        #기존 업체정보, 업종정보 결합
        c_merge = pd.merge(c_ind, c_biz, on='ent_cd', how='left')

        #reg_df(일주일 데이터)와 기존 데이터 병합
        new_df = pd.concat([c_merge,reg_df], ignore_index=False)

        # 중복 확인을 위한 컬럼 제외
        columns_to_exclude = ['x_coord', 'y_coord', 'sigungu_code', 'sigungu_nm', 
                            'induty_mid_inte', 'induty_mid_spe', 
                            'induty_mid_inte_sub', 'induty_mid_spe_sub', 
                            'se_cd', 'open_status','biz_visible','email','fax_no', 'jurirno','idx','induty_regist_no']

        # 제외할 컬럼을 사용하여 중복된 값 확인
        new_df = new_df.drop_duplicates(subset=new_df.columns.difference(columns_to_exclude), keep='first')
        # 폐업 여부
        for i, j in cess_df.iterrows():
            c_name = j['ncrGsKname']
            c_date = j['ncrGsDate']
            c_item = j['ncrItemName']

            new_df.loc[
                (new_df['ncrgskname'] == c_name) & (new_df['ncritemname'] == c_item) &
                (new_df['ncrgsdate'] <= c_date), 'open_status'
            ] = '폐업'

        for i, j in cess_df.iterrows():
            reg_no = j['ncrMasterNum']
            c_date = j['ncrGsDate']
            c_item = j['ncrItemName']    

            new_df.loc[
                (new_df['ncrmasternum'] == reg_no) & (new_df['ncritemname'] == c_item) &
                (new_df['ncrgsdate'] <= c_date), 'open_status'
            ] = '폐업'
        

        #영업정지 여부
        admi_df = admi_df[admi_df['ncrAdmiDename']=='영업정지']
        admi_df = admi_df.loc[admi_df.groupby('ncrAdmiKname')['ncrAdmiStopEdate'].idxmax()].reset_index(drop=True)

        # 종료날짜를 YYYYMMDD 형식으로 문자열로 변환
        today_date = datetime.strptime(str(eDate), "%Y%m%d")

        # 영업정지 데이터 기준으로 open_status 업데이트
        for index, row in admi_df.iterrows():
            if row['ncrAdmiDename'] == '영업정지':
                c_name = row['ncrAdmiKname']
                se_date = datetime.strptime(str(row['ncrAdmiStopEdate']), "%Y%m%d")  # 날짜를 YYYYMMDD 형식으로 변환

                # 날짜 비교
                if today_date < se_date:
                    new_df.loc[
                        (new_df['ncrgskname'] == c_name), 'open_status'
                    ] = '영업정지'

        for col in reg_df.columns:
            if col in new_df.columns:
                new_df[col] = new_df[col].astype(reg_df[col].dtype)

        # new_df['se_cd'] = new_df['se_cd'].astype('str', errors='ignore')
        new_df['x_coord'] = pd.to_numeric(new_df['x_coord'], errors='coerce') 
        new_df['y_coord'] = pd.to_numeric(new_df['y_coord'], errors='coerce')
        new_df['sigungu_code'] = pd.to_numeric(new_df['sigungu_code'], errors='coerce')

        # 업체명 (주), 주식회사 통일
        new_df['ncrgskname'] = new_df['ncrgskname'].apply(lambda x: re.sub(r'주식회사', '(주)', x).strip())

        # 중복 제거: 업체별 업종별로 ncrgsregdate 컬럼이 가장 큰 값을 가지는 행만 남기기
        new_df = new_df.loc[new_df.groupby(['ncrgskname', 'ncritemname'])['ncrgsregdate'].idxmax()].reset_index(drop=True)

        # 영업정지 데이터 기준으로 open_status 업데이트
        for index, row in admi_df.iterrows():
            company_name = row['ncrAdmiKname']
            business_reg_no = row['ncrItemregno']
            representative_name = row['ncrAdmiMaster']
            industry_name = row['ncrItemName']
            end_date = datetime.strptime(str(row['ncrAdmiStopEdate']), "%Y%m%d")  # 종료 날짜를 datetime 형식으로 변환

            f_date = datetime.strptime(eDate, "%Y%m%d") 
            f_end_date = end_date.strftime('%Y/%m/%d')

            # 영업정지 여부 판단
            if f_date < end_date:
                # 사업자등록번호와 업종명으로 매핑
                new_df.loc[
                    (new_df['ncritemregno'] == business_reg_no) &
                    (new_df['ncritemname'] == industry_name), 'open_status'
                ] = f'영업정지(~ {f_end_date})'

                # 사업자등록번호 결측치인 경우 업체명, 대표자명, 업종명으로 매핑
                new_df.loc[
                    (new_df['ncritemregno'].isna()) &
                    (new_df['ncrgskname'] == company_name) &
                    (new_df['ncrgsmaster'] == representative_name) &
                    (new_df['ncritemname'] == industry_name), 'open_status'
                ] = f'영업정지(~ {f_end_date})'

        new_df = new_df[new_df['open_status'] != '폐업'].reset_index(drop=True)

        # 업체 구분을 위한 ent_cd 생성
        # 업체명 오름차순 정렬 및 사업자등록번호 기준으로 구분
        new_df = new_df.sort_values(by=['ncrgskname', 'ncrmasternum']).reset_index(drop=True)

        # ent_cd 컬럼 생성 및 초기화
        new_df['ent_cd'] = 0

        # 사업자등록번호 기준으로 ent_cd 값 부여
        ent_cd_counter = 1
        for idx, row in new_df.iterrows():
            if pd.notna(row['ncrmasternum']):
                if idx == 0 or row['ncrmasternum'] != new_df.loc[idx - 1, 'ncrmasternum']:
                    new_df.at[idx, 'ent_cd'] = ent_cd_counter
                    ent_cd_counter += 1
                else:
                    new_df.at[idx, 'ent_cd'] = new_df.loc[idx - 1, 'ent_cd']
                    
            else:  # 사업자등록번호 결측치인 경우 업체명과 대표자명으로 구분
                if idx == 0 or not (
                    row['ncrgskname'] == new_df.loc[idx - 1, 'ncrgskname'] and
                    row['ncrgsmaster'] == new_df.loc[idx - 1, 'ncrgsmaster']
                ):
                    new_df.at[idx, 'ent_cd'] = ent_cd_counter
                    ent_cd_counter += 1
                else:
                    new_df.at[idx, 'ent_cd'] = new_df.loc[idx - 1, 'ent_cd']

        # c_ind 데이터프레임 생성
        new_c_ind_columns = ['ncrgsdate', 'ncrgsaddr', 'ncritemname', 'induty_mid_inte', 'induty_mid_spe', 'induty_mid_inte_sub', 'induty_mid_spe_sub', 'se_cd', 'open_status', 'idx', 'ent_cd']
        new_c_ind = new_df[new_c_ind_columns]

        # c_biz 데이터프레임 생성 - 동일 ent_cd 중 ncrgsdate가 가장 큰 행만 유지
        new_c_biz_columns = ['ncrareaname', 'ncrgsflag', 'ncrgskname', 'ncrgsmaster', 'ncrgsnumber', 'ncrgsreason', 'ncrgsregdate', 'ncrgsseq', 'ncritemregno', 'ncrmasternum', 'ncrofftel', 'x_coord', 'y_coord', 'sigungu_code', 'sigungu_nm', 'ent_cd']
        new_c_biz = new_df.loc[new_df.groupby('ent_cd')['ncrgsdate'].idxmax()][new_c_biz_columns].reset_index(drop=True)

        new_c_ind = new_c_ind.sort_values(by='ent_cd').reset_index(drop=True)
        new_c_ind['idx'] = np.arange(1, len(new_c_ind) + 1)

        new_c_biz.to_sql(name=f'tb_crack_down_biz', con=db_engine, if_exists='replace', schema='company', index=False)
        new_c_ind.to_sql(name=f'tb_crack_down_industry', con=db_engine, if_exists='replace', schema='company', index=False)

    except json.decoder.JSONDecodeError as e:
        print(f"JSON 디코딩 오류: {e}")
    except requests.exceptions.RequestException as e:
        print(f"요청 예외: {e}")
    except Exception as e:
        print(f"기타 오류: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='API 호출 날짜 입력')
    parser.add_argument('sDate', type=str, help='시작 날짜 (YYYY-MM-DD 형식)')
    parser.add_argument('eDate', type=str, help='종료 날짜 (YYYY-MM-DD 형식)')
    
    args = parser.parse_args()
    main(args.sDate, args.eDate)