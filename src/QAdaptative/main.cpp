#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/RectangleShape.hpp>
#include <SFML/Window/Event.hpp>

#include "imgui.h"
#include "imgui-SFML.h"

#include <torch/torch.h>

#include <iostream>
#include <random>
#include <string>

inline std::string vect2fToString(const sf::Vector2f& v) {
    return "(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ")";
}

inline void testTorch() {
    if (torch::cuda::is_available()) {
        std::cout << "CUDA est disponible! Entra�nement sur GPU." << std::endl;

        // Cr�e un tensor simple
        torch::Tensor tensor = torch::rand({ 2, 3 });

        // Affiche le tensor
        std::cout << "Tensor avant transfert sur GPU:" << std::endl;
        std::cout << tensor << std::endl;

        // Transf�re le tensor sur le GPU
        tensor = tensor.to(torch::kCUDA);

        // Affiche le tensor apr�s transfert sur GPU
        std::cout << "Tensor apr�s transfert sur GPU:" << std::endl;
        std::cout << tensor << std::endl;
    }
    else {
        std::cout << "CUDA n'est pas disponible. Entra�nement sur CPU." << std::endl;
    }
}

int main() {
    testTorch();

    sf::RenderWindow window(sf::VideoMode(1280, 720), "QAdaptative");
    window.setFramerateLimit(60);
    sf::Clock deltaClock;

    ImGui::SFML::Init(window);

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(-10, 10);
    std::uniform_real_distribution<float> dist2(0.75, 1.25);

    sf::RectangleShape rect(sf::Vector2f(50, 50));
    rect.setPosition(window.getPosition().x - rect.getSize().x / 2, window.getPosition().y - rect.getSize().y / 2);
    rect.setFillColor(sf::Color::Black);
    rect.setOutlineColor(sf::Color::White);
    rect.setOutlineThickness(2);

    sf::Vector2f direction(dist(mt), dist(mt));

    sf::Event event;
    while (window.isOpen()) {
        while (window.pollEvent(event)) {
            ImGui::SFML::ProcessEvent(event);
            switch (event.type) {
                case sf::Event::Closed:
                    window.close();
                    break;
            }
        }

        ImGui::SFML::Update(window, deltaClock.restart());

        ImGui::Begin("Variables visualizer");
        ImGui::Text(("Rect position : " + vect2fToString(rect.getPosition())).c_str());
        ImGui::Text(("Rect direction : " + vect2fToString(direction)).c_str());
        ImGui::Text(("Rect size : " + vect2fToString(rect.getSize())).c_str());
        ImGui::End();

        sf::Vector2f nextPos = rect.getPosition() + direction;
        if (nextPos.x < 0 || nextPos.x >= window.getSize().x - rect.getSize().x) {
            direction.x = -direction.x * dist2(mt);
            rect.setSize(sf::Vector2f(50 * 0.85f, 50 * 1.15f));
        }
        if (nextPos.y < 0 || nextPos.y >= window.getSize().y - rect.getSize().y) {
            direction.y = -direction.y * dist2(mt);
            rect.setSize(sf::Vector2f(50 * 1.15f, 50 * 0.85f));
        }

        rect.setPosition(nextPos);

        window.clear();
        window.draw(rect);
        ImGui::SFML::Render(window);
        window.display();
    }

    ImGui::SFML::Shutdown();
    return 0;
}