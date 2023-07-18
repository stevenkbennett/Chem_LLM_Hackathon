import os
import random
from datetime import datetime

import dash
from dash import Input, Output, callback, dcc, html
from github import Github
from requests_cache import DO_NOT_CACHE, install_cache

external_stylesheets = [
    "https://fonts.googleapis.com/css?family=Oswald|Overpass+Mono&amp;display=swap",
]

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width"}],
    title="Chemistry LLM Hackathon Leaderboard",
    external_stylesheets=external_stylesheets,
)
app.css.config.serve_locally = True

# cache GH requests to placate API
install_cache(
    cache_control=True,
    urls_expire_after={
        "*.github.com": 60,  # Placeholder expiration; overridden by Cache-Control
        "*": DO_NOT_CACHE,  # Don't cache anything other than GitHub requests
    },
)



def get_team_list():
    # define github api settings
    gh = Github(os.environ["GITHUB_ACCESS_TOKEN"])
    repo = gh.get_repo("stevenkbennett/fons_datathon_testing")
    open_prs = repo.get_pulls()
    teams = []
    for pr in open_prs:
        score = None
        time = None
        url = None
        comments = pr.get_issue_comments()
        for comment in comments:
            if "Total Points" in comment.body:
                tmp_score = int(comment.body.split()[-1].split("/")[0])
                if score is None or tmp_score > score:
                    score = tmp_score
                    time = comment.created_at.strftime('%H:%M on %b %d')
                    url = comment.html_url

        if score is not None:
            teams.append(
                {
                    "name": pr.title,
                    "handle": pr.user.login,
                    "time": time,
                    "pr_url": url,
                    "img": pr.user.avatar_url,
                    "score": score,
                }
            )
    teams = sorted(teams, key=lambda x: x["score"], reverse=True)
    for i, t in enumerate(teams):
        t["rank"] = i + 1

    team_list = []
    for team in teams:
        team_list.append(
            create_list_entry(
                team["rank"],
                team["name"],
                team["time"],
                team["img"],
                team["score"],
                team["pr_url"],
            )
        )

    return team_list


def create_list_entry(rank, name, time, img, score, pr_url):
    if rank == 1:
        extra_place = " u-text--dark u-bg--yellow"
        extra_kudos = " u-text--yellow"
        emoji = "🏆"
    elif rank == 2:
        extra_place = " u-text--dark u-bg--teal"
        extra_kudos = " u-text--teal"
        emoji = "⭐️"
    elif rank == 3:
        extra_place = " u-text--dark u-bg--orange"
        extra_kudos = " u-text--orange"
        emoji = "🔥"
    else:
        extra_place = ""
        extra_kudos = ""
        emoji = random.choice(("👏", "👍", "🙌", "🤩", "💯"))

    return html.Li(
        [
            html.Div(
                [
                    html.Div(
                        [rank],
                        className="c-flag c-place u-bg--transparent" + extra_place,
                    ),
                    html.Div(
                        [
                            html.Img(src=img, className="c-avatar c-media__img"),
                            html.Div(
                                [
                                    html.Div([name], className="c-media__title"),
                                    html.A(
                                        time,
                                        href=pr_url,
                                        className="c-media__link u-text--small",
                                        target="_blank",
                                    ),
                                ],
                                className="c-media__content",
                            ),
                        ],
                        className="c-media",
                    ),
                    html.Div(
                        [html.Div([html.Strong(score), emoji], className="u-mt--8")],
                        className="u-text--right c-kudos" + extra_kudos,
                    ),
                ],
                className="c-list__grid",
            )
        ],
        className="c-list__item",
    )


def get_leaderboard():
    return html.Li(
        [
            html.Div(
                [
                    html.Div(["Rank"], className="u-text--left u-text--small u-text--medium"),
                    html.Div(
                        ["Team Name"],
                        className="u-text--left u-text--small u-text--medium",
                    ),
                    html.Div(
                        ["Score"],
                        className="u-text--right u-text--small u-text--medium",
                    ),
                ],
                className="c-list__grid",
            ),
        ]
        + get_team_list(),
        className="c-list__item",
    )


app.layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [html.H3("Chemistry Hackathon Leaderboard")],
                    className="c-card__header",
                ),
                html.Div(
                    [html.Ul([get_leaderboard()], className="c-list", id="list")],
                    className="c-card__body",
                ),
            ],
            className="c-card",
        ),
        dcc.Interval(
            id="interval-component",
            interval=60 * 1000,  # in milliseconds
            n_intervals=0,
        ),
    ],
    className="l-wrapper",
)


@callback(Output("list", "children"), Input("interval-component", "n_intervals"))
def refresh_leaderboard(n):
    return get_leaderboard()


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", debug=False, port=8050)
