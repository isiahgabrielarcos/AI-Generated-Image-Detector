// Background service worker for AI Image Detector Extension

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "analyzeImage",
    title: "Analyze with AI Detector",
    contexts: ["image"]
  });
  console.log("AI Image Detector Extension installed");
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId !== "analyzeImage" || !tab?.id) return;

  chrome.storage.local.set({ lastImage: info.srcUrl });

  chrome.tabs.sendMessage(tab.id, {
    action: "OPEN_OVERLAY_AND_ANALYZE",
    imageUrl: info.srcUrl
  });
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "open_dashboard") {
    chrome.tabs.create({ url: chrome.runtime.getURL("dashboard.html") });
  }

  if (request.action === "open_overlay") {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      const tab = tabs?.[0];
      if (!tab?.id) return;
      chrome.tabs.sendMessage(tab.id, { action: "OPEN_OVERLAY" });
    });
  }

  if (request.action === "analyze_image") {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      const tab = tabs?.[0];
      if (!tab?.id) return;
      chrome.tabs.sendMessage(tab.id, {
        action: "OPEN_OVERLAY_AND_ANALYZE",
        imageUrl: request.imageData
      });
    });
  }

  return true;
});

console.log("AI Image Detector background service worker loaded");
