```mermaid
flowchart TD

    SCRAPE[Scrape from Zillow]

    subgraph CLEAN_ZILLOW[Clean/Process Zillow Data]
        ZRAW[(Raw Zillow Data <br/> raw_sold.csv)]
        ZRAW --> ZCLEAN[Clean Data <br/> clean_data.py]
        ZCLEAN --> ZCLEANED[(Cleaned Zillow Data <br/> cleaned_sold.csv)]
    end

    subgraph DOWNLOAD_IMAGES_GRAPH[Download Images]
        ZCLEANED --> SPLIT[Split Data <br/> split_data.py]
        SPLIT --> IMAGE_LINKS[(Image Links <br/> image_links.csv)]
        IMAGE_LINKS --> DOWNLOAD_IMAGES[Download Images <br/> download_images.py]
    end

    subgraph DOWNLOAD_PLUTO[Download PLUTO]
        ZCLEANED --> GET_BBL[Geoclient V2]
        GET_BBL --> PLUTO[PLUTO <br/> 64uk-42ks]
    end

    SCRAPE --> ZRAW
    DOWNLOAD_IMAGES --> IMAGES
    PLUTO --> PLUTO_DATA

    subgraph DATA_SRC[Data Sources]
        STRUCT_ATTR[(Structural<br/>Attributes)]
        IMAGES[(Images)]
        PRICE[(Sale Price)]
        PLUTO_DATA[(Pluto Data <br/> pluto_data.csv)]
    end

    ZCLEANED --> STRUCT_ATTR
    ZCLEANED --> PRICE

    IMAGES --> ROOM_CNN

    subgraph IMAGES_PROC[Process Images]
        ROOM_CNN[Room Classifier <br/> CNN]
        ROOM_CNN --> IN_ROOM_SET{Room is in <br/> Room Set}
        IN_ROOM_SET --> |Yes| LLM_IMG_PROMPT[Prompt LLM with correspodning room type prompt]
        IN_ROOM_SET --> |No| LLM_CHOOSE_ROOM
        LLM_IMG_PROMPT --> LLM_AGREES{LLM agrees with <br/> room type}
        LLM_AGREES --> |No| LLM_CHOOSE_ROOM[LLM Chooses Correct Room and correct prompt]
        LLM_AGREES --> |Yes| LLM_SCORE[GPT Gives Luxury Score <br/> and Reason]
        LLM_CHOOSE_ROOM --> |Room is in room set| LLM_SCORE
    end

    LLM_SCORE --> AGGR_PHOTOS[Average across same rooms within each property]
    AGGR_PHOTOS --> IMG_MATRIX[(Luxury Scores per property<br/>Matrix)]

    subgraph POI_PROC[Process POIs]
        PLUTO_DATA --> CREATE_CIRCLES[Create Circle of 600m radius around each property]
        CREATE_CIRCLES --> POI_DATE_AGGR[Group properties into <br/> sold month]
        POI_DATE_AGGR --> OHSOME[Get POI counts per category in each month]
    end

    OHSOME --> POI_MATRIX[(POI Scores per Property <br/> Matrix)]

    JOIN_TABLES[Join Tables]
    STRUCT_ATTR --> JOIN_TABLES
    IMG_MATRIX --> JOIN_TABLES
    POI_MATRIX --> JOIN_TABLES
    PLUTO_DATA --> JOIN_TABLES
    JOIN_TABLES --> ALL_FEATURES[(All Combined Features <br/> Matrix)]

    ALL_FEATURES --> XGBOOST[XGBoost <br/> Regression]
    XGBOOST --> PRICE_PRED([Predicted Sale Price])
    XGBOOST --> LOSS[Loss]
    PRICE --> LOG_PRICE[Log Price]
    LOG_PRICE --> LOSS
```