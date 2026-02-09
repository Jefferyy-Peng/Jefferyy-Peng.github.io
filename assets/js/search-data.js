// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-publications",
          title: "publications",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/publications/";
          },
        },{id: "nav-projects",
          title: "projects",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/projects/";
          },
        },{id: "nav-teaching",
          title: "teaching",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/teaching/";
          },
        },{id: "post-generalization-through-variance-in-diffusion-models",
        
          title: "Generalization Through Variance in Diffusion Models",
        
        description: "Paper summary for &quot;Generalization through Variance in Diffusion Models&quot;",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/diffusion-generalization-variance/";
          
        },
      },{id: "post-how-diffusion-models-work",
        
          title: "How Diffusion Models Work",
        
        description: "A learning notebook for diffusion models",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/diffusion-model/";
          
        },
      },{id: "books-the-godfather",
          title: 'The Godfather',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/the_godfather/";
            },},{id: "news-i-obtained-my-b-e-degree-at-xidian-univesity",
          title: 'I obtained my B.E. degree at Xidian Univesity.',
          description: "",
          section: "News",},{id: "news-start-to-pursue-my-master-s-degree-at-columbia-univerisy",
          title: 'Start to pursue my master’s degree at Columbia Univerisy.',
          description: "",
          section: "News",},{id: "news-start-to-do-research-at-liinc-lab",
          title: 'Start to do research at LIINC lab.',
          description: "",
          section: "News",},{id: "news-i-obtained-my-m-s-degree-at-columbia-university",
          title: 'I obtained my M.S. degree at Columbia University.',
          description: "",
          section: "News",},{id: "news-one-paper-was-accepted-by-joss-2024",
          title: 'One paper was accepted by JOSS 2024.',
          description: "",
          section: "News",},{id: "news-start-my-ph-d-journey-at-deepreal-lab-ud-newark",
          title: 'Start my Ph.D. journey at DeepREAL Lab@UD, Newark.',
          description: "",
          section: "News",},{id: "news-one-paper-was-accepted-to-icml-2025",
          title: 'One paper was accepted to ICML 2025!',
          description: "",
          section: "News",},{id: "projects-data-driven-mri-postprocessing",
          title: 'Data-Driven MRI Postprocessing',
          description: "A data-driven, deep-learning based MRI postprocessing framework to improve medical image analysis",
          section: "Projects",handler: () => {
              window.location.href = "/projects/1_project/";
            },},{id: "projects-project-4",
          title: 'project 4',
          description: "another without an image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/4_project/";
            },},{id: "projects-project-5",
          title: 'project 5',
          description: "a project with a background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/5_project/";
            },},{id: "projects-project-6",
          title: 'project 6',
          description: "a project with no image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/6_project/";
            },},{id: "projects-project-9",
          title: 'project 9',
          description: "another project with an image 🎉",
          section: "Projects",handler: () => {
              window.location.href = "/projects/9_project/";
            },},{
        id: 'social-cv',
        title: 'CV',
        section: 'Socials',
        handler: () => {
          window.open("/assets/pdf/Yunxiang_Peng_CV_Jan_2026.pdf", "_blank");
        },
      },{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%79%78%70%65%6E%67%63%73@%75%64%65%6C.%65%64%75", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/yunxiang-peng", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=1MVsftMAAAAJ", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
