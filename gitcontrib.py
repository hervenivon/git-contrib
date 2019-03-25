#!/usr/bin/env python3

import argparse
import numpy as np
import os
import random
import re
import requests
import sys
from sty import fg, rs

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(here, 'gitcontrib'))
import ga  # noqa
import manual  # noqa

COLORS = ["#ebedf0",
          "#c6e48b",
          "#7bc96f",
          "#239a3b",
          "#196127"]
NBCLASS = len(COLORS)

SHAPE = (53, 7)
FLATSHAPE = 53 * 7


def rgb2hex(r, g, b):
    return f'#{r:02x}{g:02x}{b:02x}'


def hex2rgb(hx):
    return (int(hx[1:3], 16), int(hx[3:5], 16), int(hx[5:7], 16))


def cprint(txt='', end='\n', color='#ffffff'):
    r, g, b = hex2rgb(color)
    print(fg(r, g, b) + txt + rs.fg, end=end)


class Images:
    SpaceInvaders = np.array(
        [[0, 0, 4, 0, 0, 0, 0, 0, 4, 0, 0],
         [0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0],
         [0, 0, 4, 4, 4, 4, 4, 4, 4, 0, 0],
         [0, 4, 4, 0, 4, 4, 4, 0, 4, 4, 0],
         [4, 0, 4, 4, 4, 4, 4, 4, 4, 0, 4],
         [4, 0, 4, 0, 0, 0, 0, 0, 4, 0, 4],
         [0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0]], dtype=int)

    def getCalendar():
        blk = np.zeros((7, 21), dtype=int)
        res = np.concatenate((blk, Images.SpaceInvaders), axis=1)
        res = np.concatenate((res, blk), axis=1)
        return res.T.reshape(FLATSHAPE)

    def getCalendar2():
        blk3 = np.zeros((7, 3), dtype=int)
        blk1 = np.zeros((7, 1), dtype=int)
        res = np.concatenate((blk3, Images.SpaceInvaders), axis=1)
        res = np.concatenate((res, blk1), axis=1)
        res = np.concatenate((res, Images.SpaceInvaders), axis=1)
        res = np.concatenate((res, blk1), axis=1)
        res = np.concatenate((res, Images.SpaceInvaders), axis=1)
        res = np.concatenate((res, blk1), axis=1)
        res = np.concatenate((res, Images.SpaceInvaders), axis=1)
        res = np.concatenate((res, blk3), axis=1)
        return res.T.reshape(FLATSHAPE)


class Calendar:
    CHARACTER = '██'

    def empty_calandar():
        return np.zeros(shape=SHAPE, dtype=int)

    def print_random():
        res = ''
        for _ in range(7):
            for _ in range(53):
                r, g, b = hex2rgb(COLORS[random.randint(0, 4)])
                res += fg(r, g, b) + Calendar.CHARACTER + rs.fg
            res += '\n'
        print(res)

    def stringify_elt(e):
        r, g, b = hex2rgb(COLORS[int(e)])
        return fg(r, g, b) + '██' + rs.fg

    def random_string_numpy():
        a = np.random.randint(5, size=(7, 53))
        stringify = np.vectorize(Calendar.stringify_elt)
        stringify_all = np.vectorize(Calendar.stringify_elt)
        s = stringify_all(a)

        return '\n'.join([''.join(x) for x in s])

    def random_string_python():
        res = ''
        for _ in range(7):
            for _ in range(53):
                r, g, b = hex2rgb(COLORS[random.randint(0, 4)])
                res += fg(r, g, b) + Calendar.CHARACTER + rs.fg
            res += '\n'
        return res

    def normalized_calendar(self):
        tmp = self.calendar
        if np.count_nonzero(tmp) > 0:
            tmp = np.ceil(tmp * (NBCLASS - 1) /
                          tmp.max()).astype(int)
        return tmp.reshape(SHAPE).T

    def shaped_calendar(self):
        return self.calendar.reshape(SHAPE).T

    def __init__(self, first_day, calendar=None):
        assert isinstance(first_day, str)
        assert re.match(r'^\d{4}-\d{2}-\d{2}$', first_day)

        self.first_day = first_day
        if calendar is not None:
            self.calendar = calendar
        else:
            self.calendar = np.random.randint(5, size=FLATSHAPE)

    def __str__(self):
        stringify = np.vectorize(Calendar.stringify_elt)
        stringify_all = np.vectorize(Calendar.stringify_elt)

        s = stringify_all(self.normalized_calendar())

        return '\n'.join([''.join(x) for x in s])


def run_query(query, variables, token):
    """ A simple function to use requests.post to make the GithubAPI call. """
    headers = {'Authorization': 'Bearer %s' % token}
    request = requests.post('https://api.github.com/graphql',
                            json={'query': query, 'variables': variables},
                            headers=headers)
    if request.status_code == 200:
        return request.json()
    else:
        raise Exception('Query failed to run by returning code '
                        'of {}. {}'.format(request.status_code, query))


def get_contributions(username, token):
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
        'login': username
    }

    return run_query(query, variables, token)


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


def generate_argparse():
    parser = argparse.ArgumentParser(
        description='Update a given github user contributions wall',
        prog='git-contrib.py')
    parser.add_argument(
        'username', nargs=1,
        help='The github username for which the contribution wall'
             ' is being changed')
    parser.add_argument(
        '-t', '--token', nargs='?',
        help='The github bearer token to use to gather the historic'
             ' contribution of a user')

    return parser


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


def print_random():
    for _ in range(7):
        for _ in range(52):
            cprint('██', end='', color=COLORS[random.randint(0, 4)])
        print('')


def main():
    parser = generate_argparse()
    args = parser.parse_args()

    results = get_contributions(args.username[0], args.token)

    first_day = first_date(results)
    contributionsasnp = githubres2nparray(results)

    actual_calendar = Calendar(first_day, calendar=contributionsasnp)
    expected_calendar = Calendar(first_day, calendar=Images.getCalendar2())

    print('current contributions: %d' % contributionsasnp.sum())
    print(actual_calendar)

    print('target')
    print(expected_calendar)

    newcontrib = manual.getOptimizedIndividual(
                        expected_calendar=expected_calendar.calendar,
                        actual_calendar=actual_calendar.calendar,
                        shape=SHAPE,
                        flatshape=FLATSHAPE,
                        nbclass=NBCLASS)
    print('manual new contributions: %d' % newcontrib.sum())
    print(Calendar(first_day, newcontrib + actual_calendar.calendar))

    newcontrib, nbgen = ga.getOptimizedIndividual(
                           expected_calendar=expected_calendar.calendar,
                           actual_calendar=actual_calendar.calendar,
                           shape=SHAPE,
                           flatshape=FLATSHAPE,
                           nbclass=NBCLASS)

    print('genetic new contributions optimization: %d with'
          ' %d generation' % (newcontrib.sum(), nbgen))
    print(Calendar(first_day, newcontrib + actual_calendar.calendar))


if __name__ == "__main__":
    main()
