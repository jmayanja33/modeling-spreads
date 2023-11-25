from fastapi import APIRouter
from fastapi.responses import JSONResponse

from cse6242_project.api.datamodels import BettingLineInput, ScoreInput
from cse6242_project.utilities import load_model
from cse6242_project.utilities import WeeklyRanking


rf_score_model = load_model('rf_regressor_score.pkl')
router = APIRouter(prefix='/infer')
weekly_rankings = WeeklyRanking()


@router.get('/betting_line')
def get_betting_line(data=BettingLineInput):
    pass


@router.get('/score')
def get_betting_line(data: ScoreInput):
    weekly_ranking_data = weekly_rankings.get_rankings(
        data.team_name)
    model_output = rf_score_model.predict(weekly_ranking_data)
    return JSONResponse({'score': model_output[0]})
