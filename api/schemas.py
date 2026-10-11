"""Request / response models shared by the routers."""
import math
from typing import List, Literal, Optional

from fastapi import HTTPException, status
from pydantic import AnyHttpUrl, BaseModel, Field, model_validator

# class MlflowRun(BaseModel):
#     run_id: str
#     experiment_name: str
#     experiment_id: str
#     metrics: Dict[str, float]
#     params: Dict[str, str]

class InputData(BaseModel):
    sqr_m:             float = Field(..., ge=5.0,  le=500.0, description="Apartment size (sqm)")
    # Deposit / key money are often not decided when a listing is being priced, and they are
    # near-leaky (deposit == rent in ~58% of listings). Omit both to get a prediction that
    # does not lean on them; /predict and /challenger_predict support that, other routes don't.
    rei_price:         Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Key money (万円). Optional; omit if not yet known")
    shikikin:          Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Security deposit (万円). Optional; omit if not yet known")
    maintenence_price: float = Field(..., ge=0.0,  le=50.0,  description="Monthly maintenance fee (万円)")
    year_built:        float = Field(..., ge=0.0,  le=100.0, description="Years since construction")
    floor:             float = Field(..., ge=0.0,  le=60.0,  description="Floor number (0 = ground)")
    eki_walk:          float = Field(..., ge=0.0,  le=120.0, description="Walk to nearest station (minutes)")
    # Optional categorical features — improves prediction accuracy when provided
    ku_name:           Optional[str] = Field(default=None, description="Tokyo ward (e.g. '渋谷区', '新宿区')")
    apartment_type:    Optional[str] = Field(default=None, description="Room layout (e.g. '1K', '2LDK', 'ワンルーム')")
    house_type:        Optional[str] = Field(default=None, description="Building type (e.g. '賃貸マンション', '賃貸アパート')")
    address:           Optional[str] = Field(default=None, max_length=100, description="Street address to the block (e.g. '東京都渋谷区猿楽町１'). Optional; used by the no-deposit/key-money models to look at nearby rents. An unfamiliar address is geocoded with the GSI service (disable with GEO_LIVE_GEOCODE=0)")
    nearest_station:   Optional[str] = Field(default=None, max_length=50, description="Nearest station name without 駅 (e.g. '渋谷'). Optional; used by the no-deposit/key-money models")
    nearest_line:      Optional[str] = Field(default=None, max_length=50, description="Line serving that station (e.g. '東急東横線'). Optional; used by the no-deposit/key-money models")

    @model_validator(mode="after")
    def check_finite(self) -> "InputData":
        for name, value in self.__dict__.items():
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                raise ValueError(f"Field '{name}' must be a finite number, got {value}")
        return self


TERMS_FIELDS = ("rei_price", "shikikin")


def _require_terms(item: "InputData") -> dict:
    """Dict form of *item* for models that cannot handle missing deposit / key money."""
    d = item.dict()
    missing = [f for f in TERMS_FIELDS if d.get(f) is None]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{', '.join(missing)} required for this endpoint "
                   "(only /predict, /challenger_predict, /compare_predictions and A/B routes accept it omitted)",
        )
    return d


class RetrainData(BaseModel):
    url: AnyHttpUrl
    wait_time_min: int = Field(default=1, ge=1, le=60)
    wait_time_max: int = Field(default=5, ge=1, le=120)
    pages: tuple = Field(default=(0, 50))
    model_to_retrain: Optional[List[Literal["challenger", "passed", "tabpfn", "tabnet", "torch"]]] = Field(
        default=["challenger", "passed"]
    )

    @model_validator(mode="after")
    def validate_retrain(self) -> "RetrainData":
        if self.wait_time_min >= self.wait_time_max:
            raise ValueError("wait_time_min must be less than wait_time_max")
        p = self.pages
        if (not isinstance(p, (list, tuple)) or len(p) != 2
                or not all(isinstance(x, int) for x in p)
                or p[0] < 0 or p[1] < 0 or p[0] >= p[1]):
            raise ValueError("pages must be a 2-element tuple [start, end] with 0 <= start < end")
        return self


class WardRetrainData(BaseModel):
    """Schema for multi-ward scrape-and-retrain requests."""
    ward_codes: Optional[List[str]] = Field(
        default=None,
        description=(
            "List of Suumo sc codes (e.g. ['13104', '13113']).  "
            "Defaults to all 23 Tokyo special wards."
        ),
    )
    pages_per_ward: int = Field(
        default=10, ge=1, le=100,
        description="Pages to scrape per ward (~30 listings/page).",
    )
    wait_time_min: int = Field(default=2, ge=1, le=60)
    wait_time_max: int = Field(default=5, ge=2, le=120)
    model_to_retrain: Optional[List[Literal["challenger", "passed", "tabpfn", "tabnet", "torch"]]] = Field(
        default=["challenger", "passed"]
    )

    @model_validator(mode="after")
    def validate_wait_times(self) -> "WardRetrainData":
        if self.wait_time_min >= self.wait_time_max:
            raise ValueError("wait_time_min must be less than wait_time_max")
        return self


class EnsemblePredictRequest(BaseModel):
    input_data: InputData
    method: Literal["weighted_average", "simple_average", "stacking", "best_model"] = Field(
        default="weighted_average", description="Ensemble method to use"
    )


class BatchPredictRequest(BaseModel):
    input_data_list: List[InputData] = Field(..., min_length=1, max_length=100)
    method: Literal["ensemble_weighted_average", "simple_average", "best_model"] = Field(
        default="ensemble_weighted_average", description="Prediction method to use for all inputs"
    )


class ABTestCreateRequest(BaseModel):
    experiment_name: str  = Field(..., min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_\-]+$")
    control_model:   str  = Field(default="passed")
    treatment_model: str  = Field(default="challenger")
    traffic_split:   float = Field(default=0.1, ge=0.0, le=1.0)
    shadow_mode:     bool  = Field(default=False)
    min_samples:     int   = Field(default=100,  ge=10,    le=100_000)
    confidence_level: float = Field(default=0.95, ge=0.5,  le=0.999)
    min_effect_size:  float = Field(default=0.05, ge=0.001, le=1.0)


class ABTestPredictRequest(BaseModel):
    experiment_name: str
    input_data: InputData
    user_id: Optional[str] = None
    force_model: Optional[str] = None


class TorchPredictRequest(BaseModel):
    input_data: InputData
    model_name: Literal["tokyo_rent_torch", "tokyo_rent_tabpfn", "tokyo_rent_tabnet"] = Field(
        description="Name of the PyTorch/pretrained model to use"
    )


class TorchBatchPredictRequest(BaseModel):
    input_data_list: List[InputData] = Field(..., min_length=1, max_length=100)
    model_name: Literal["tokyo_rent_torch", "tokyo_rent_tabpfn", "tokyo_rent_tabnet"] = Field(
        description="Name of the PyTorch/pretrained model to use for batch predictions"
    )


class ExplainRequest(BaseModel):
    input_data: InputData
    model_name: Literal["challenger", "passed", "tabpfn", "tabnet"] = Field(
        default="challenger", description="Model to explain"
    )
    method: Literal["shap", "lime"] = Field(default="shap", description="Explanation method")


class GroundTruthItem(BaseModel):
    prediction_id: str = Field(..., max_length=64, description="`prediction_id` from the /predict response")
    actual_value: float = Field(..., gt=0, description="Actual monthly rent in 万円")


class GroundTruthRequest(BaseModel):
    labels: List[GroundTruthItem] = Field(..., min_length=1, max_length=1000)
