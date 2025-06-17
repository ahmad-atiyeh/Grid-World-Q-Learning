#include <iostream>
#include <vector>
#include <map>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <limits>

using namespace std;

int N;
bool sliding_enabled;

const int PLAYER = 1;
const int ENEMY = 2;
const int GOAL = 3;
const int EMPTY = 0;

vector<vector<int>> board;
pair<int, int> player_pos, goal_pos;
vector<pair<int, int>> enemies;
map<pair<int, int>, map<string, double>> q_table;

const double alpha = 0.1;
const double gamma = 0.9;
const double epsilon = 0.1;

vector<string> actions = { "U", "D", "L", "R" };
map<string, pair<int, int>> action_delta = {
    {"U", {-1, 0}},
    {"D", {1, 0}},
    {"L", {0, -1}},
    {"R", {0, 1}}
};

void clear_screen() {
#ifdef _WIN32
    system("cls");
#else
    system("clear");
#endif
}

vector<string> get_valid_actions(int x, int y) {
    vector<string> valid;
    for (auto& a : actions) {
        int nx = x + action_delta[a].first;
        int ny = y + action_delta[a].second;
        if (nx >= 0 && nx < N && ny >= 0 && ny < N) {
            valid.push_back(a);
        }
    }
    return valid;
}

pair<int, int> move(pair<int, int> state, string action) {
    int x = state.first;
    int y = state.second;
    int dx = action_delta[action].first;
    int dy = action_delta[action].second;

    int step = (sliding_enabled && (rand() % 100 < 30)) ? 2 : 1;

    for (int i = 0; i < step; i++) {
        int nx = x + dx;
        int ny = y + dy;
        if (nx >= 0 && nx < N && ny >= 0 && ny < N) {
            x = nx;
            y = ny;
        }
        else {
            break;
        }
    }
    return { x, y };
}

int get_reward(pair<int, int> state) {
    if (state == goal_pos) return 100;
    for (auto& enemy : enemies) {
        if (state == enemy) return -100;
    }
    return sliding_enabled ? -2 : -1;
}

void init_q(pair<int, int> state) {
    if (q_table.find(state) == q_table.end()) {
        for (auto& a : actions) {
            q_table[state][a] = 0.0;
        }
    }
}

string choose_action(pair<int, int> state) {
    init_q(state);
    vector<string> valid = get_valid_actions(state.first, state.second);
    if (valid.empty()) return "";

    if ((double)rand() / RAND_MAX < epsilon) {
        return valid[rand() % valid.size()];
    }
    else {
        string best = valid[0];
        for (auto& a : valid) {
            if (q_table[state][a] > q_table[state][best]) {
                best = a;
            }
        }
        return best;
    }
}

void train(int episodes = 1000) {
    for (int ep = 0; ep < episodes; ep++) {
        pair<int, int> state = player_pos;
        while (state != goal_pos) {
            init_q(state);
            string action = choose_action(state);
            if (action == "") break;
            pair<int, int> new_state = move(state, action);
            int reward = get_reward(new_state);
            init_q(new_state);

            double max_q = -1e9;
            for (auto& entry : q_table[new_state]) {
                if (entry.second > max_q) max_q = entry.second;
            }

            q_table[state][action] += alpha * (reward + gamma * max_q - q_table[state][action]);
            if (reward == -100) break;
            state = new_state;
        }
    }
}

void print_board() {
    for (int i = 0; i < board.size(); i++) {
        for (int j = 0; j < board[0].size(); j++) {
            cout << board[i][j] << " ";
        }
        cout << endl;
    }
}

void print_q_table(int level) {
    cout << "\nQ-Table for Level " << level << " (2D Grid of best actions):\n";
    vector<vector<string>> best_actions(board.size(), vector<string>(board[0].size(), "--"));
    vector<vector<double>> best_values(board.size(), vector<double>(board[0].size(), -1e9));

    for (auto& entry : q_table) {
        pair<int, int> state = entry.first;
        string best_action = "";
        double best_value = -1e9;
        for (auto& q : entry.second) {
            if (q.second > best_value) {
                best_value = q.second;
                best_action = q.first;
            }
        }
        best_actions[state.first][state.second] = best_action;
        best_values[state.first][state.second] = best_value;
    }

    cout << "\nBest Action per Cell:\n";
    for (int i = 0; i < best_actions.size(); i++) {
        for (int j = 0; j < best_actions[0].size(); j++) {
            cout << best_actions[i][j] << "\t";
        }
        cout << endl;
    }

    cout << "\nQ-Value per Cell:\n";
    for (int i = 0; i < best_values.size(); i++) {
        for (int j = 0; j < best_values[0].size(); j++) {
            if (best_values[i][j] == -1e9)
                cout << "----\t";
            else
                cout << best_values[i][j] << "\t";
        }
        cout << endl;
    }
}

string choose_greedy_action(pair<int, int> state) {
    init_q(state);
    vector<string> valid = get_valid_actions(state.first, state.second);
    if (valid.empty()) return "";
    string best = valid[0];
    for (auto& a : valid) {
        if (q_table[state][a] > q_table[state][best]) {
            best = a;
        }
    }
    return best;
}

void follow_best_path() {
    pair<int, int> state = player_pos;
    vector<string> path;
    while (state != goal_pos) {
        string action = choose_greedy_action(state);
        if (action == "") break;
        pair<int, int> new_state = move(state, action);
        path.push_back(action);
        if (new_state == state) break;
        state = new_state;
    }
    cout << "\nAgent path: ";
    for (auto& step : path) cout << step << " ";
    cout << endl;
}

int main() {
    srand((unsigned int)time(0));

    cout << "Enable sliding (1 = yes, 0 = no)? ";
    int slide_choice;
    cin >> slide_choice;
    sliding_enabled = (slide_choice == 1);

    for (int level = 1; level <= 4; level++) {
        cout << "\nPress Enter to start level " << level << "..." << endl;
        cin.ignore();
        cin.get();
        clear_screen();

        int random_rows = 4 + rand() % 3;
        int random_cols = 4 + rand() % 3;
        N = max(random_rows, random_cols);
        board = vector<vector<int>>(random_rows, vector<int>(random_cols, EMPTY));
        q_table.clear();
        enemies.clear();

        int px = rand() % board.size();
        int py = rand() % board[0].size();
        int gx = rand() % board.size();
        int gy = rand() % board[0].size();
        while (gx == px && gy == py) {
            gx = rand() % board.size();
            gy = rand() % board[0].size();
        }
        player_pos = { px, py };
        goal_pos = { gx, gy };

        cout << "Level " << level << " - Board size: rows = " << board.size() << ", cols = " << board[0].size() << endl;
        cout << "Goal is located at: (" << goal_pos.first << ", " << goal_pos.second << ")\n";

        board[player_pos.first][player_pos.second] = PLAYER;
        board[goal_pos.first][goal_pos.second] = GOAL;

        cout << "\nInitial Board:\n";
        print_board();

        int enemy_count;
        cout << "Enter number of enemies: ";
        cin >> enemy_count;

        for (int i = 0; i < enemy_count; i++) {
            int x, y;
            while (true) {
                cout << "Enter enemy " << i + 1 << " position (x y): ";
                cin >> x >> y;
                if ((x != player_pos.first || y != player_pos.second) &&
                    (x != goal_pos.first || y != goal_pos.second) &&
                    x >= 0 && x < board.size() && y >= 0 && y < board[0].size()) {
                    enemies.push_back({ x, y });
                    board[x][y] = ENEMY;
                    break;
                }
                else {
                    cout << "Invalid position. Try again.\n";
                }
            }
        }

        train(1000);
        print_q_table(level);
        follow_best_path();
        cout << "\nFinal Board State:\n";
        print_board();
    }

    return 0;
}