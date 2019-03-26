import numpy as np
import requests


class Github:
    def __init__(self, username, token, repo=None):
        self.username = username
        self.repo = repo
        self.token = token

    def _run_query(self, query, variables):
        '''
        A simple function to use requests.post to make the GithubAPI call.
        '''

        headers = {'Authorization': 'Bearer %s' % self.token}
        request = requests.post('https://api.github.com/graphql',
                                json={'query': query, 'variables': variables},
                                headers=headers)
        if request.status_code == 200:
            return request.json()
        else:
            raise Exception('Query failed to run by returning code '
                            'of {}. {}'.format(request.status_code, query))

    def get_contributions(self):
        '''
        The GraphQL query (with a few additional bits included) itself defined
        as a multi-line string.
        '''
        query = """
        query GetContributions($login: String!){
            user(login: $login) {
                contributionsCollection {
                contributionCalendar {
                    months {
                        firstDay
                        name
                        totalWeeks
                        year
                    }
                    totalContributions
                    weeks {
                    contributionDays {
                            color
                            contributionCount
                            date
                            weekday
                        }
                    }
                }
                }
            }
        }
        """
        variables = {
            'login': self.username
        }

        return self._run_query(query, variables)


def first_date(githubres):
    user_data = githubres['data']['user']
    contributions_collection = user_data['contributionsCollection']
    months = contributions_collection['contributionCalendar']['months']

    return months[0]['firstDay']


def githubres2nparray(githubres):
    user_data = githubres['data']['user']
    contributions_collection = user_data['contributionsCollection']
    weeks = contributions_collection['contributionCalendar']['weeks']

    flat_weeks = []
    for week in weeks:
        for day in week['contributionDays']:
            flat_weeks.append(day['contributionCount'])

    a = np.array(flat_weeks.copy(), dtype=int)
    a.resize(7*53)

    return a


def print_results(github_results):
    (weeks) = (github_results['data']
                             ['user']
                             ['contributionsCollection']
                             ['contributionCalendar']
                             ['weeks'])

    for wd in range(7):
        for w in range(len(weeks)):
            contributions = weeks[w]['contributionDays']
            try:
                cprint('██', end='', color=contributions[wd]['color'])
            except:
                print('  ', end='')
        print('')


def _print_random():
    for _ in range(7):
        for _ in range(52):
            cprint('██', end='', color=COLORS[random.randint(0, 4)])
        print('')
