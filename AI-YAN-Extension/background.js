// Background service worker for Artify Extension

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "analyzeImage",
    title: "Analyze with AI-YAN",
    contexts: ["image"]
  });
  console.log("Artify Extension installed");
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

console.log("Artify background service worker loaded");
