#pragma once
#include <bits/stdint-uintn.h>
#include <cmath>
#include <glm/fwd.hpp>
#include <memory>
#define GLFW_INCLUDE_VULKAN
#include <cstring>
#include <set>
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <vulkan/vulkan_core.h>
#include <cstdint>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstdlib>
#include <iterator>
#include <optional>
#include <fstream>
#include <array>
#include <glm/gtc/matrix_transform.hpp>

#include <chrono>
struct Vertex {
    glm::vec2 pos;
    glm::vec3 color;
    float shouldMove;
    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);
        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);
        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32_UINT;
        attributeDescriptions[2].offset = offsetof(Vertex, shouldMove);
        return attributeDescriptions;
    }

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return bindingDescription;
    }
};

struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};
extern std::vector<Vertex> curVertices;
extern glm::vec2 curOffset; 
extern  glm::vec4 arr;
const std::vector<Vertex> vertices = {
    {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, 0},
    {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f},1},
    {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f},2},
    {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}, 3}
};
const std::vector<Vertex> vertices1 = {
    {{-0.5f, -0.75f}, {1.0f, 0.0f, 0.0f},0},
    {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, 1},
    {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}, 2},
    {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}, 3}
};

const std::vector<uint16_t> indices = {
    0, 1, 2, 2, 3, 0
};
const int MAX_FRAMES_IN_FLIGHT = 2;


static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }
    size_t fileSize = (size_t) file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    return buffer;
}

inline VkResult CreateDebugUtilsMessengerEXT(VkInstance instance,
                                      const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                      const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {

    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

    return VK_FALSE;
}

inline void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

class MainVulkan {
public:
    void run();
private:

    const std::vector<const char*> validationLayers = {
        "VK_LAYER_LUNARG_standard_validation"
    };
    const std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };
    struct SwapChainSupportDetails {
        VkSurfaceCapabilitiesKHR capabilities;
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;
    };
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = reinterpret_cast<MainVulkan*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }
 static void keyCallback(GLFWwindow *w, int key, int scancode, int action,
                          int mods) {
    auto app =   reinterpret_cast<MainVulkan *>(
            glfwGetWindowUserPointer(w));

    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
     // app->handleKeyPress(key);
    } else if (action == GLFW_RELEASE) {
      app->releaseKey();
    }
  }

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif
    GLFWwindow * window;
    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;
    VkQueue graphicsQueue;
    VkSurfaceKHR surface;
    VkQueue presentQueue;
    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    std::vector<VkImageView> swapChainImageViews;
    std::vector<VkFramebuffer> swapChainFramebuffers;
    std::vector<VkCommandBuffer> commandBuffers;
VkDescriptorPool descriptorPool;
std::vector<VkDescriptorSet> descriptorSets;
std::vector<VkDescriptorSet> descriptorSetsVec;
std::vector<VkDescriptorSet> descriptorSetsArr;



    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    VkRenderPass renderPass;
    std::vector<VkPipeline> graphicsPipelines;
    VkCommandPool commandPool;
    std::vector<VkBuffer> vertexBuffers;
    std::vector<VkDeviceMemory> vertexBuffersMemories;
    std::vector<VkBuffer> indicesBuffers;
    std::vector<VkDeviceMemory> indicesBuffersMemories;
    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
std::vector<VkBuffer> uniformBuffersVec;
    std::vector<VkDeviceMemory> uniformBuffersMemoryVec;
std::vector<VkBuffer> uniformBuffersArr;
    std::vector<VkDeviceMemory> uniformBuffersMemoryArr;



//    VkBuffer vertexBuffer;
    //  VkDeviceMemory vertexBufferMemory;
    // VkBuffer indexBuffer;
    // VkDeviceMemory indexBufferMemory;
    size_t currentFrame = 0;
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorSetLayout descriptorSetLayoutVec;




    VkPipelineLayout pipelineLayout;
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    std::vector<VkFence> imagesInFlight;
    bool framebufferResized = false;
bool keyPressed = false;
bool flip = true; 
    const uint32_t WIDTH = 800;
    const uint32_t HEIGHT = 600;
    void initVulkan();
    void initWindow();
    void mainLoop();
    void cleanup();
    void createVulkan();
    bool checkValidationLayerSupport();
    std::vector<const char*> getRequiredExtensions();
    void setupDebugMessenger();
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);
    void pickPhysicalDevice();
    bool isDeviceSuitable(VkPhysicalDevice device);
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
    void createLogicalDevice();
    void createSurface();
    bool checkDeviceExtensionSupport(VkPhysicalDevice device);
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) ;
    void createSwapChain() ;
    void      createImageViews();
    void     createGraphicsPipeline();
    VkShaderModule createShaderModule(const std::vector<char>& code);
    void createRenderPass();
    void createFramebuffers();
    void createCommandPool();
    void createCommandBuffers();
    void drawFrame();
    void createSemaphores();
    void createVertexBuffers(std::vector<Vertex> v);
    void     createIndexBuffer();
    void recreateSwapChain();
    void cleanupSwapChain();
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                      VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) ;
    void passCommand();
    void createDescriptorSetLayout();
    void     createUniformBuffers();
    void updateUniformBuffer(uint32_t currentImage);
    void updateUniformBuffer1(uint32_t currentImage, bool flip);
void createDescriptorPool();      
void createDescriptorSets() ;
void initKeyBindings();
void releaseKey();    
};
